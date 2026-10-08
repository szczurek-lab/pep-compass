import logging
from abc import ABC
from collections import defaultdict
from copy import deepcopy

import Levenshtein
import numpy as np
import torch
from joblib import Parallel, delayed, parallel_backend

from pep_compass.local_enumeration.mutation_enumerator import \
    MutationEnumerator
from pep_compass.local_enumeration.sampling_walker import \
    SamplingWalker
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import \
    HydrAMPEncoderDecoder
from pep_compass.utils.sequence_utils import translate_generated_peptide

logger = logging.getLogger(__name__)

class LocalEnumerator(ABC):

    def local_enumeration(self, center_sequence: str) -> set[str]:
        """
        Local enumeration method to be implemented by subclasses.
        This method should return a set containing sequences in the local neighborhood of the center_sequence.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class SamplingMutationLocalEnumerator(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        sampling_walker: SamplingWalker,
        mutation_enumerator: MutationEnumerator,
        walker_trajectories_number: int = 10,
        time_walk_budget: float = 0.1,
        max_neighbour_levenstein: int | None = 4,
        device: str = "cpu",
    ):
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.sampling_walker = sampling_walker
        self.mutation_enumerator = mutation_enumerator
        self.walker_trajectories_number = walker_trajectories_number
        self.time_walk_budget = time_walk_budget
        self.max_neighbour_levenstein = max_neighbour_levenstein
        self.device = device

        self.max_neighbour_levenstein = max_neighbour_levenstein or 25

    def local_enumeration(self, center_peptide) -> set[str]:
        neighbor_peptides = set()

        with torch.no_grad():
            initial_latent_position = self.encoder_decoder.encode_peptides(
                [center_peptide]
            )[0]

        for trajectory_iter in range(self.walker_trajectories_number):

            current_latent_position = initial_latent_position
            time_walk = 0.0
            current_peptide = center_peptide
            walker_step = 0

            while time_walk < self.time_walk_budget:

                new_latent_position, step_info = self.sampling_walker.step(
                    current_latent_position
                )
                adjusted_time_step = step_info["adjusted_time_step"]
                U = step_info["U"].cpu().detach().numpy()
                S = step_info["S"].cpu().detach().numpy()

                mutated_peptides = self.mutation_enumerator.mutate(
                    current_peptide, U=U, S=S
                )

                with torch.no_grad():
                    current_peptide = self.encoder_decoder.decode_peptides(
                        new_latent_position
                    )[0]
                current_latent_position = new_latent_position
                time_walk += adjusted_time_step
                walker_step += 1

                new_neighbor_peptides = [
                    peptide
                    for peptide in mutated_peptides
                    if Levenshtein.distance(peptide, center_peptide)
                    <= self.max_neighbour_levenstein
                ]

                neighbor_peptides.update(new_neighbor_peptides)

                logger.info(
                    f"Trajectory {trajectory_iter} Step {walker_step} Time {time_walk} / {self.time_walk_budget} Levenstain {Levenshtein.distance(current_peptide, center_peptide)}: Found {len(neighbor_peptides)} peptides ."
                )
                if (
                    Levenshtein.distance(current_peptide, center_peptide)
                    > self.max_neighbour_levenstein
                ):
                    logger.info(
                        f"Reached {Levenshtein.distance(current_peptide, center_peptide)} distance. Stopping trajectory."
                    )
                    break

        return neighbor_peptides


class EuclideanWalkerLocalEnumeratorWithAmbientDistance(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        walker_trajectories_number: int,
        max_walker_ambient_distance: float,
        max_neighbour_levenstein: int = None,
        time_step: float = 0.1,
        device: str = "cpu",
    ):
        self.encoder_decoder = encoder_decoder
        self.walker_trajectories_number = walker_trajectories_number
        self.max_walker_ambient_distance = max_walker_ambient_distance
        self.time_step = time_step
        self.max_neighbour_levenstein = max_neighbour_levenstein
        self.device = device
        self.max_neighbour_levenstein = max_neighbour_levenstein or 25

    def local_enumeration(self, center_peptide: str) -> set[str]:

        neighbor_peptides = set()

        initial_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]
        initial_ambient_point = self.encoder_decoder.decoder_forward(
            initial_latent_point, softmax_and_flatten=True
        )

        logger.info(
            f"Time step: {self.time_step}, Total ambient distance: {self.max_walker_ambient_distance }"
        )

        for i in range(self.walker_trajectories_number):

            walker_ambient_distance = 0.0
            current_latent_point = initial_latent_point.clone()
            previous_ambient_position = initial_ambient_point.clone()

            while walker_ambient_distance <= self.max_walker_ambient_distance:  # = 16.0

                # Euclidean walker robi 20 kroków tak żęby czas wynosił 0.25. CZyli jego time_step = 0.25/20; spatal_step = sqrt(time_step) = sqrt(0.25/20)
                # 0.25 * 64 = 16

                # Generate a random direction in the latent space
                direction = torch.randn(
                    initial_latent_point.shape[-1], device=self.device
                )
                direction = (
                    direction
                    / torch.norm(direction)
                    * np.sqrt(initial_latent_point.shape[-1])  # srqt(64)
                )

                current_latent_point += direction * self.time_step**0.5

                decoder_output = self.encoder_decoder.decoder_forward(
                    current_latent_point, softmax_and_flatten=False
                )
                current_peptide = translate_generated_peptide(decoder_output)

                current_ambient_position = torch.softmax(
                    decoder_output / self.encoder_decoder.temp, dim=-1
                ).flatten()

                ambient_dist = torch.dist(
                    previous_ambient_position, current_ambient_position, p=2
                )
                walker_ambient_distance += ambient_dist.item()

                logger.info(
                    f"Trajectory {i} Ambient distance {walker_ambient_distance:.4f} / {self.max_walker_ambient_distance:.4f} Levenshtein {Levenshtein.distance(current_peptide, center_peptide)}"
                )

                if (
                    Levenshtein.distance(current_peptide, center_peptide)
                    > self.max_neighbour_levenstein
                ):
                    logger.info(
                        f"Reached {Levenshtein.distance(current_peptide, center_peptide)} distance. Stopping trajectory."
                    )
                    break
                neighbor_peptides.add(current_peptide)

        return neighbor_peptides


class EuclideanWalkerLocalEnumerator(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        walker_trajectories_number: int,
        walker_time: float,
        max_neighbour_levenstein: int = None,
        time_step: float = 0.1,
        batch_size: int = 1000,
        device: str = "cpu",
    ):
        self.encoder_decoder = encoder_decoder
        self.walker_trajectories_number = walker_trajectories_number
        self.walker_time = walker_time
        self.device = device
        self.time_step = time_step
        self.max_neighbour_levenstein = max_neighbour_levenstein or 25
        self.batch_size = batch_size

    def local_enumeration(self, center_peptide) -> set[str]:
        neighbor_peptides = set()

        initial_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]

        number_of_steps = int(self.walker_time / self.time_step)

        logger.info(
            f"Number of steps: {number_of_steps}, Time step: {self.time_step}, Total time: {self.walker_time}"
        )

        normal_sample = torch.randn(
            (
                self.walker_trajectories_number,
                number_of_steps,
                initial_latent_point.shape[-1],
            ),
            device=self.device,
        )

        normal_sample = (
            normal_sample
            / torch.norm(normal_sample, dim=-1, keepdim=True)
            # * torch.sqrt(Sigma z enkodera)
            * self.time_step**0.5
            * np.sqrt(initial_latent_point.shape[-1])
        )

        latent_trajectories = initial_latent_point + torch.cumsum(normal_sample, dim=1)

        batched_latent_trajectories = latent_trajectories.reshape(
            self.walker_trajectories_number * number_of_steps,
            initial_latent_point.shape[-1],
        )

        with torch.no_grad():
            decoded_peptides = self.encoder_decoder.decode_peptides(
                batched_latent_trajectories, batch_size=self.batch_size
            )

        neighbor_peptides.add(decoded_peptides)

        return neighbor_peptides


class NormalSamplingLocalEnumerator(LocalEnumerator):

    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        max_neighbour_levenstein: int = None,
        number_of_samples: int = 1000,
        sampling_temperature: float = 1.0,
        batch_size: int = 5000,
        device: str = "cpu",
    ):
        self.max_neighbour_levenstein = max_neighbour_levenstein or 25
        self.batch_size = batch_size
        self.encoder_decoder = encoder_decoder
        self.device = device

        self.sampling_temperature = sampling_temperature
        self.number_of_samples = number_of_samples

    def local_enumeration(self, center_peptide) -> set[str]:

        mean, std = self.encoder_decoder.encode_peptides_with_std([center_peptide])
        std = torch.exp(std / 2)
        mean, std = mean[0], std[0]  # (64, ), (64, )

        sampled_latent_points = (
            torch.randn((self.number_of_samples, mean.shape[0]), device=self.device)
            * std
            * self.sampling_temperature
            + mean
        )  # (64, )

        generated_peptides = self.encoder_decoder.decode_peptides(
            sampled_latent_points, batch_size=self.batch_size
        )

        neighbour_peptides = {
            peptide
            for peptide in generated_peptides
            if Levenshtein.distance(peptide, center_peptide)
            <= self.max_neighbour_levenstein
        }

        return neighbour_peptides


class MutationLocalEnumerator(LocalEnumerator):
    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        mutation_generator: MutationEnumerator,
        max_neighbour_levenstein: int = None,
        device: str = "cpu",
    ):
        self.encoder_decoder = encoder_decoder
        self.mutation_generator = mutation_generator
        self.device = device
        self.max_neighbour_levenstein = max_neighbour_levenstein

        if self.max_neighbour_levenstein is None:
            self.max_neighbour_levenstein = 25

    def local_enumeration(self, center_peptide, **kwargs) -> set[str]:

        with torch.no_grad():
            # Encode the center peptide to get the latent point
            center_latent_point = self.encoder_decoder.encode_peptides(
                [center_peptide]
            )[0]

        with torch.no_grad():
            jacobian = self.encoder_decoder.decoder_jacobian(center_latent_point)
            logger.debug(f"jacobian device: {jacobian.device}")
            U, S, V = torch.linalg.svd(jacobian, full_matrices=False)

            logger.debug(f"U, S, V device: {U.device}, {S.device}, {V.device}")

        mutated_peptides = self.mutation_generator.mutate(center_peptide, U=U, S=S)

        neighbor_peptides = {
            peptide
            for peptide in mutated_peptides
            if Levenshtein.distance(peptide, center_peptide)
            <= self.max_neighbour_levenstein
        }

        return neighbor_peptides


# TODO: It was an attempt to parallelize the walker trajectories. Probably it can be useful in the future.
# class MultiWalkerLocalEnumerator(LocalEnumerator):
#     def __init__(
#         self,
#         encoder_decoder: HydrAMPEncoderDecoder,
#         walker_factory: RiemannianWalkerFactory,
#         walker_trajectories_number: int,
#         walker_time: float,
#         max_neighbour_levenstein: int = None,
#         max_neighbour_levenstein_frac: float = None,
#         device: str = "cpu",
#     ):
#         super().__init__(max_neighbour_levenstein, max_neighbour_levenstein_frac)
#         self.encoder_decoder = encoder_decoder
#         self.walker_factory = walker_factory
#         self.walker_trajectories_number = walker_trajectories_number
#         self.walker_time = walker_time
#         self.device = device

#     def reset(self):
#         pass

#     def walk_trajectory(
#         self,
#         center_peptide,
#         initial_latent_point,
#         initial_latent_velocity,
#         max_neighbour_levenstein,
#     ) -> set[str]:
#         peptides_in_trajectory = set()

#         walker = self.walker_factory.create()
#         step_peptide = center_peptide

#         walker_step = 0
#         while walker.time < self.walker_time:
#             walker.step()

#             current_latent_point = walker.current_step.latent_position

#             step_peptide = translate_generated_peptide(
#                 self.encoder_decoder.decoder_forward(
#                     current_latent_point.to(self.device),
#                     softmax_and_flatten=False,
#                 )
#             )
#             logger.info(
#                 f"Step {walker_step} Time {walker.time} / {self.walker_time} Levenstain {Levenshtein.distance(step_peptide, center_peptide)}"
#             )
#             if (
#                 Levenshtein.distance(step_peptide, center_peptide)
#                 > max_neighbour_levenstein
#             ):
#                 logger.info(
#                     f"Reached {Levenshtein.distance(step_peptide, center_peptide)} distance. Stopping trajectory."
#                 )
#                 break

#             walker_step += 1

#         return peptides_in_trajectory

#     def local_enumeration(self, center_peptide, **kwargs) -> set[]:

#         # TODO: refactor this part
#         if self.max_neighbour_levenstein is None:
#             max_neighbour_levenstein = len(center_peptide)
#         else:
#             max_neighbour_levenstein = self.max_neighbour_levenstein
#         if self.max_neighbour_levenstein_frac is not None:
#             max_neighbour_levenstein = int(
#                 self.max_neighbour_levenstein_frac * len(center_peptide)
#             )

#         neighbor_peptides = set()

#         initial_latent_point = self.encoder_decoder.encode_peptides([center_peptide])[0]
#         initial_latent_velocity = torch.zeros_like(initial_latent_point)

#         neighborhood = Neigborhood(
#             center_peptide=center_peptide,
#             trajectories=[],
#             # score=self.scored_peptides[center_peptide],
#         )

#         with parallel_backend("loky"):
#             trajectories = Parallel(n_jobs=4)(
#                 delayed(self.walk_trajectory)(
#                     center_peptide,
#                     initial_latent_point,
#                     initial_latent_velocity,
#                     max_neighbour_levenstein,
#                 )
#                 for _ in range(self.walker_trajectories_number)
#             )

#         for trajectory in trajectories:
#             # Add the trajectory to the neighborhood
#             neighborhood.trajectories.append(trajectory)

#             # Collect unique peptides from the trajectory
#             for step in trajectory.steps:
#                 neighbor_peptides.add(step.peptide)

#             logger.info(f"Trajectory completed with {len(trajectory.steps)} steps.")

#         return neighborhood
