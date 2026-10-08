import logging

import Levenshtein
import numpy as np
import pybktree
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from cachetools import LRUCache
from gpytorch.mlls import ExactMarginalLogLikelihood

from pep_compass.local_enumeration.local_enumerator import LocalEnumerator
from pep_compass.optimization.black_box.toxipep_black_box import AbstractBlackBox
from pep_compass.optimization.lebo.fingerprints import Map4Fingerprint
from pep_compass.optimization.lebo.kernel import TanimotoSimilarityKernel
from pep_compass.optimization.optimizer import AbstractOptimizer
from pep_compass.utils.utils import Timer, set_seed

logger = logging.getLogger(__name__)


class LocalEnumerationBayesianOptimizer(AbstractOptimizer):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        local_enumerator: LocalEnumerator,
        device: str = "cpu",
        evaluations_per_iteration: int = 3,
        levenstain_diversity_threshold: int = 2,
        initial_peptides_number: int = 10,
        turbo_success_tolerance: int = 100,
        turbo_failure_tolerance: int = 1,
        turbo_length_init: int = 2,
        turbo_length_min: int = 2,
        turbo_length_max: int = 2,
        turbo_increase_step: int = 1,
        turbo_decrease_step: int = 1,
        acquisition_batch_size: int = 32,
        standardize: bool = False,
        best_as_center: bool = False,
    ):
        super().__init__(black_box)

        self.local_enumerator = local_enumerator
        self.device = device
        self.map4 = Map4Fingerprint(input_type="fasta", chiral=False)
        self.evaluations_per_iteration = evaluations_per_iteration
        self.levenstain_diversity_threshold = levenstain_diversity_threshold
        self.initial_peptides_number = initial_peptides_number
        self.acquisition_batch_size = acquisition_batch_size

        self.turbo_success_tolerance = turbo_success_tolerance
        self.turbo_failure_tolerance = turbo_failure_tolerance
        self.turbo_distance_init = turbo_length_init
        self.turbo_distance_min = turbo_length_min
        self.turbo_distance_max = turbo_length_max
        self.turbo_increase_step = turbo_increase_step
        self.turbo_decrease_step = turbo_decrease_step
        self.standardize = standardize
        self.best_as_center = best_as_center

        self.scored_peptides = {}
        self.not_scored_peptides_bktree = pybktree.BKTree(Levenshtein.distance)
        self.not_scored_peptides_set = set()
        self.peptide_features = LRUCache(maxsize=5_000_000)

        self.timer = Timer()

        if self.black_box.maximize:
            self.scorer = lambda peptides: -black_box(np.array(peptides))[:, 0]
        else:
            self.scorer = lambda peptides: black_box(np.array(peptides))[:, 0]

    def _turbo_filter(self):
        """
        Filter not scored peptides based on the trust region distance.
        """

        for distance in range(self.trust_region_distance, 25):
            filtered = self.not_scored_peptides_bktree.find(
                self.the_best_peptide, distance
            )
            exist_not_scored_peptide = False
            for _, peptide in filtered:
                if peptide in self.not_scored_peptides_set:
                    exist_not_scored_peptide = True
                    break
            if exist_not_scored_peptide:
                break

        return [
            peptide for _, peptide in filtered if peptide not in self.scored_peptides
        ]

    def _bayesian_optimization(self, max_scorer_calls) -> str:
        # At the beggining of the search, score few random peptides to initialize the surrogate model
        if len(self.scored_peptides) == 1:
            self._initialize_scored_peptides()

        with self.timer("extract train features"):
            # Make train set from all scored peptides
            train_features = self._extract_features(self.scored_peptides.keys())
            train_scores = list(self.scored_peptides.values())

            train_X = torch.tensor(
                train_features, dtype=torch.float64, requires_grad=False
            ).to(self.device)
            train_Y = (
                torch.tensor(train_scores, dtype=torch.float64, requires_grad=False)
                .reshape(-1, 1)
                .to(self.device)
            )

        with self.timer("fit gp"):
            # Fit the GP model
            gp = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                covar_module=TanimotoSimilarityKernel(),
                # mean_module=None,
                # input_transform=Normalize(d=train_X.shape[1]),
                # TODO: what is the purpose of this transform?
                outcome_transform=Standardize(m=1) if self.standardize else None,
            ).to(self.device)

            mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(self.device)

            fit_gpytorch_mll(mll)
            gp.eval()

        with self.timer("Construct acquisition function"):
            logEI = LogExpectedImprovement(
                model=gp, best_f=train_Y.min(), maximize=False
            ).to(self.device)

        with self.timer("turbo filter"):
            # Filter test peptides based on the trust region distance
            test_peptides = self._turbo_filter()

            logger.info(f"Number of test peptides: {len(test_peptides)}")

            if len(test_peptides) == 0:
                raise ValueError("No test peptides left to evaluate.")

        with self.timer("extract test features"):
            test_peptides_features = self._extract_features(test_peptides)

            logger.info(f"Extracted features for {len(test_peptides)} test peptides")

            test_X = (
                torch.tensor(
                    test_peptides_features, dtype=torch.float64, requires_grad=False
                )
                .cpu()
                .unsqueeze(1)
            )
            logger.info(f"Test features device: {test_X.device}")

        with self.timer("compute logEI"):
            # Compute the acquisition function for the not scored peptides
            with torch.no_grad():
                test_logEI = []
                for i in range(0, len(test_X), self.acquisition_batch_size):
                    batch = test_X[i : i + self.acquisition_batch_size].to(self.device)
                    logger.debug(f"Test features shape: {batch.shape}")

                    # TODO: this is hack. Because There is something wrong with Standarize and Tanimoto Kernel which squeezes when batch=1.
                    is_one_batch = batch.shape[0] == 1
                    if is_one_batch:
                        batch = batch.repeat(2, 1, 1)

                    with torch.no_grad():
                        compute_logEI = logEI(batch)

                    if is_one_batch:
                        compute_logEI = compute_logEI[:1]

                    test_logEI.append(compute_logEI.cpu())
                    logger.debug(
                        f"Computed logEI for batch {i // self.acquisition_batch_size + 1}"
                    )
                    # del batch
                test_logEI = torch.cat(test_logEI)

            logger.info(
                f"Acquisition function computed for {len(test_peptides)} test peptides"
            )

            logger.info("\n")

        with self.timer("robot"):
            peptides_to_evaluate = []
            for _ in range(
                min(
                    self.evaluations_per_iteration,
                    max_scorer_calls - self.black_box_calls,
                )
            ):
                if len(test_peptides) == 0:
                    break

                best_improvement_index = test_logEI.argmax()

                best_improvement_peptide = test_peptides[best_improvement_index]
                peptides_to_evaluate.append(best_improvement_peptide)

                logger.info(
                    f"Expected Improvement ({best_improvement_peptide}): {test_logEI[best_improvement_index].item()}"
                )

                remaining_inices = []
                for i, peptide in enumerate(test_peptides):
                    if (
                        Levenshtein.distance(peptide, best_improvement_peptide)
                        > self.levenstain_diversity_threshold
                    ):
                        remaining_inices.append(i)

                test_peptides = [test_peptides[i] for i in remaining_inices]
                test_logEI = test_logEI[remaining_inices]

                logger.info(f"{len(test_peptides)} test peptides left")

        with self.timer("evaluate"):
            # Evaluate the most promising peptides
            best_scores = self.scorer(peptides_to_evaluate)
            self.black_box_calls += len(peptides_to_evaluate)

            for peptide, score in zip(peptides_to_evaluate, best_scores):
                # Remove the peptides to evaluate from the not scored peptides
                self.not_scored_peptides_set.remove(peptide)

                # Add the peptides to evaluate to the scored peptides
                self.scored_peptides[peptide] = score

                logger.info(f"Evaluated peptide {peptide} with score {score}")

        # Get the best peptide and its score
        best_score_index = best_scores.argmin()
        best_score = best_scores[best_score_index]
        best_peptide = peptides_to_evaluate[best_score_index]

        if best_score < self.the_best_score:
            self.the_best_peptide = best_peptide
            self.the_best_score = best_score

            self.turbo_success_counter += 1
            self.turbo_failure_counter = 0
        else:
            self.turbo_failure_counter += 1
            self.turbo_success_counter = 0

        # Update the trust region distance
        if self.turbo_success_counter >= self.turbo_success_tolerance:
            self.trust_region_distance = max(
                self.trust_region_distance - self.turbo_decrease_step,
                self.turbo_distance_min,
            )
            logger.info(
                f"Trust region distance increased to {self.trust_region_distance}"
            )
            self.turbo_success_counter = 0

        if self.turbo_failure_counter >= self.turbo_failure_tolerance:
            self.trust_region_distance = min(
                self.trust_region_distance + self.turbo_increase_step,
                self.turbo_distance_max,
            )
            logger.info(
                f"Trust region distance decreased to {self.trust_region_distance}"
            )
            self.turbo_failure_counter = 0

        # Print interesting details about the best improvement peptide
        logger.info("\n")
        logger.info(f"Best improvement peptide: {best_peptide}")
        logger.info(f"Score: {best_score}")
        logger.info(f"Trust region distance: {self.trust_region_distance}")
        logger.info(f"Times: {self.timer}")
        self.timer.reset()

        # logger.info(f"Filtered test peptides: {len(test_peptides)}")
        logger.info("\n")

        logger.info(f"Before freeing {torch.cuda.memory_allocated() / 1024**2} MB")
        logger.info(f"Before freeing {torch.cuda.memory_reserved() / 1024**2} MB")
        del gp, mll, logEI
        torch.cuda.empty_cache()
        logger.info(f"After freeing {torch.cuda.memory_allocated() / 1024**2} MB")
        logger.info(f"After freeing {torch.cuda.memory_reserved() / 1024**2} MB")

        return best_peptide

    def _extract_features(self, peptides) -> np.ndarray:
        logger.info(
            f"Peptides in cache {len(self.peptide_features)} before extraction."
        )

        extracted_features = np.zeros((len(peptides), self.map4._dimensions))

        # Separate new peptides
        new_peptides = [p for p in peptides if p not in self.peptide_features]

        logger.info(f"Computing features for {len(new_peptides)} new peptides.")

        # Batch compute features for new peptides
        if new_peptides:
            new_features = self.map4(new_peptides)
            for peptide, feature in zip(new_peptides, new_features):
                self.peptide_features[peptide] = feature  # stored in LRU cache

        for i, peptide in enumerate(peptides):
            extracted_features[i] = self.peptide_features[peptide]

        logger.info(f"Peptides in cache {len(self.peptide_features)} after extraction.")
        return extracted_features

    def _initialize_scored_peptides(self):
        initial_peptides = list(self.not_scored_peptides_bktree)
        np.random.shuffle(initial_peptides)
        initial_peptides = initial_peptides[: self.initial_peptides_number]

        initial_scores = self.scorer(initial_peptides)
        self.black_box_calls += len(initial_peptides)

        for peptide, score in zip(initial_peptides, initial_scores):
            self.scored_peptides[peptide] = score
            self.not_scored_peptides_set.remove(peptide)
            if score < self.the_best_score:
                self.the_best_peptide = peptide
                self.the_best_score = score

    def _initialize_optimization(self, starting_point, rng_seed=None):
        if rng_seed is not None:
            set_seed(rng_seed)

        self.turbo_success_counter = 0
        self.turbo_failure_counter = 0
        self.trust_region_distance = self.turbo_distance_init

        with torch.no_grad():
            starting_peptide_score = self.scorer([starting_point]).item()

        self.scored_peptides = {starting_point: starting_peptide_score}
        self.not_scored_peptides_bktree = pybktree.BKTree(Levenshtein.distance)
        self.not_scored_peptides_set = set()
        self.peptide_features.clear()
        self._extract_features([starting_point])

        self.black_box_calls = 0
        self.the_best_peptide = starting_point
        self.the_best_score = self.scored_peptides[starting_point]

        self.timer.reset()

    def optimize(
        self,
        evaluation_budget: int,
        starting_point: str,
        rng_seed: int | None = None,
    ):

        self._initialize_optimization(starting_point, rng_seed)

        current_center_peptide = starting_point

        while self.black_box_calls < evaluation_budget:
            if self.best_as_center:
                current_center_peptide = self.the_best_peptide

            with self.timer("Local Search"):
                local_candidate_set = self.local_enumerator.local_enumeration(
                    current_center_peptide
                )

            with self.timer("Update not scored peptides"):
                peptides_to_add = set(local_candidate_set) - set(
                    self.scored_peptides.keys()
                )

                len_before = len(self.not_scored_peptides_set)
                for peptide in peptides_to_add:
                    if peptide not in self.not_scored_peptides_set:
                        self.not_scored_peptides_bktree.add(peptide)
                self.not_scored_peptides_set.update(peptides_to_add)

                logger.info(
                    f"Added {len(self.not_scored_peptides_set) - len_before} new peptides to not scored peptides."
                )

            current_center_peptide = self._bayesian_optimization(evaluation_budget)

            logger.info(
                f"Best peptide: {self.the_best_peptide} with score {self.the_best_score}"
            )
            logger.info(f"Scorer calls: {self.black_box_calls} / {evaluation_budget}")
            logger.info(
                f"Number of not scored peptides: {len(self.not_scored_peptides_set)}"
            )
