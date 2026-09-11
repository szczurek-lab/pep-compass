"""Local peptide enumeration with an isolated SORBES trajectory stream."""

from __future__ import annotations

from typing import Literal

import torch

from pep_compass.data.optimization import CandidateBatch, ObjectField, TensorField
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.step import Step
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)
TrajectoryExecution = Literal["sequential", "batched"]

_TRANSIENT_WALKER_FIELDS = (
    "walker.singular_values",
    "walker.left_vectors",
    "walker.adjusted_time_step",
    "walker.tangent_space",
    "tangent_geometry",
    "point_id",
    "tracking.trajectory_id",
    "tracking.trajectory_index",
    "tracking.rng_stream_seed",
    "tracking.trajectory_step",
)


class LocalEnumeration(Step):
    """Collect local mutations without feeding them back into SORBES.

    The operation implements the control flow of Local Enumeration. For every
    input seed and trajectory it first emits filtered MUTANG candidates at the
    initial point. It then advances SORBES one step at a time, emits the decoded
    walk point, and emits a separately filtered MUTANG batch. Only the walk
    point continues to the next iteration.

    Global selection, deduplication, oracle evaluation, and Bayesian
    optimization remain ordinary steps placed after this operation.

    :param walker: One-step trajectory transformation, normally SORBES.
    :param mutation_generator: Local mutation expansion, normally MUTANG.
    :param filters: Ordered filter flow applied to each local mutation batch.
    :param trajectories: Independent trajectories started from every seed.
    :param trajectory_execution: ``sequential`` reference execution or
        ``batched`` accelerator-oriented execution.
    :param iterations: Fixed steps per trajectory, mutually exclusive with
        ``walk_time``.
    :param walk_time: Accumulated adjusted SORBES time per trajectory, mutually
        exclusive with ``iterations``.
    :param include_walk_points: Include decoded SORBES points in the result.
    """

    def __init__(
        self,
        *,
        walker: Step,
        mutation_generator: Step,
        filters: Step,
        trajectories: int,
        iterations: int | None,
        walk_time: float | None,
        include_walk_points: bool = True,
        trajectory_execution: TrajectoryExecution = "batched",
    ) -> None:
        if trajectories < 1:
            raise ValueError("Local-enumeration trajectories must be positive.")
        if trajectory_execution not in {"sequential", "batched"}:
            raise ValueError(
                "Local-enumeration trajectory execution must be sequential or batched."
            )
        if (iterations is None) == (walk_time is None):
            raise ValueError(
                "Local enumeration requires exactly one of iterations or walk_time."
            )
        if iterations is not None and iterations < 1:
            raise ValueError("Local-enumeration iterations must be positive.")
        if walk_time is not None and walk_time <= 0:
            raise ValueError("Local-enumeration walk time must be positive.")
        self.walker = walker
        self.mutation_generator = mutation_generator
        self.filters = filters
        self.trajectories = trajectories
        self.trajectory_execution = trajectory_execution
        self.iterations = iterations
        self.walk_time = walk_time
        self.include_walk_points = include_walk_points
        requirement = getattr(mutation_generator, "geometry_requirement", None)
        contract = getattr(walker, "geometry_contract", None)
        if getattr(mutation_generator, "requires_local_enumeration", False):
            if requirement is None or contract is None or not contract.satisfies(requirement):
                raise ValueError(
                    "LocalEnumeration walker geometry does not satisfy MUTANG shared geometry."
                )

    def precompute(self, context: OptimizationContext) -> None:
        """Precompute each reusable child exactly once."""
        self.walker.precompute(context.enter_step(self.walker.name))
        self.mutation_generator.precompute(
            context.enter_step(self.mutation_generator.name)
        )
        self.filters.precompute(context.enter_step(self.filters.name))

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Execute private trajectories and collect their candidate emissions."""
        if self.trajectory_execution == "batched":
            return self._execute_batched(batch, context)
        return self._execute_sequential(batch, context)

    def _execute_sequential(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Execute one trajectory at a time as a correctness reference."""
        # Local enumeration
        ## Keep each trajectory private to prevent mutation-tree expansion
        emissions: list[CandidateBatch] = []
        trajectory_index = 0
        for seed_index in range(len(batch)):
            seed = batch.select([seed_index])
            for replica_index in range(self.trajectories):
                if context.state.stop_requested:
                    break
                trajectory_context = context.enter_branch(
                    f"seed_{seed_index:05d}_trajectory_{replica_index:05d}",
                    trajectory_index,
                )
                trajectory_index += 1
                seed = self._attach_trajectory_identity(
                    seed,
                    trajectory_context,
                    trajectory_index - 1,
                )
                emissions.extend(self._run_trajectory(seed, trajectory_context))

        if not emissions:
            return batch.select([])
        result = CandidateBatch.concatenate(emissions)
        logger.info(
            "Local enumeration seeds=%s trajectories_per_seed=%s "
            "output_candidates=%s.",
            len(batch),
            self.trajectories,
            len(result),
        )
        return result

    def _execute_batched(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Advance all active trajectories in accelerator-sized batches.

        Trajectory multiplicity is represented in the leading batch dimension.
        SORBES, decoder geometry and MUTANG therefore receive one tensor batch
        instead of repeated one-row calls. With a walk-time bound, a Boolean
        active set removes completed trajectories between steps.
        """
        # Deterministic trajectory RNG stream
        ## Isolate SORBES Torch draws from unrelated pipeline operations
        batched_context = context.enter_branch("batched_trajectories", 0)
        device = batch.latent_origins.device
        cuda_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            if batched_context.seed is not None:
                torch.manual_seed(batched_context.seed)
            return self._execute_batched_stream(batch, batched_context)

    def _execute_batched_stream(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Execute one isolated deterministic RNG stream for a trajectory batch."""
        # Trajectory initialization
        ## Repeat every seed into an independent logical trajectory
        parent_indices = torch.arange(
            len(batch), device=batch.latent_origins.device
        ).repeat_interleave(self.trajectories)  # (B * T,)
        current = batch.repeat_from_parents(parent_indices)
        current = self._attach_batched_trajectory_identity(current, context)
        current = current.with_field(
            "local_enumeration.center_sequence",
            ObjectField(current.sequences),
        )
        elapsed = torch.zeros(
            len(current),
            dtype=current.latent_origins.dtype,
            device=current.latent_origins.device,
        )  # (B * T,)
        emissions: list[CandidateBatch] = []
        iteration = 0
        # Batched SORBES trajectory evolution
        ## Keep only unfinished trajectories in the next accelerator call
        while len(current) and self._batch_should_continue(iteration, context):
            iteration_context = context.enter_iteration(iteration)
            current = self.walker(current, iteration_context)
            increments = self._time_increments(current)  # (T_active,)
            if self.include_walk_points:
                emissions.append(current.without_fields(_TRANSIENT_WALKER_FIELDS))
            emissions.append(self._generate_and_filter(current, iteration_context))
            context.state.record_iteration(len(current))

            elapsed = elapsed + increments
            iteration += 1
            current = current.with_field(
                "tracking.trajectory_step",
                TensorField(
                    current.fields["tracking.trajectory_step"].values + 1
                ),
            )
            if self.walk_time is not None:
                active = torch.nonzero(
                    elapsed < self.walk_time,
                    as_tuple=False,
                ).flatten()  # (T_next,)
                current = current.select(active)
                elapsed = elapsed.index_select(0, active)

        if not emissions:
            return batch.select([])
        result = CandidateBatch.concatenate(emissions)
        logger.info(
            "Local enumeration execution=batched seeds=%s "
            "trajectories_per_seed=%s output_candidates=%s.",
            len(batch),
            self.trajectories,
            len(result),
        )
        return result

    @staticmethod
    def _attach_trajectory_identity(
        batch: CandidateBatch,
        context: OptimizationContext,
        trajectory_index: int,
    ) -> CandidateBatch:
        """Attach replay identity to one sequential trajectory."""
        device = batch.latent_origins.device
        seed = -1 if context.seed is None else context.seed
        identifier = "/".join(context.scope.path)
        result = batch.with_field(
            "tracking.trajectory_id",
            ObjectField([identifier]),
        )
        result = result.with_field(
            "tracking.trajectory_index",
            TensorField(torch.tensor([trajectory_index], device=device)),
        )
        result = result.with_field(
            "tracking.rng_stream_seed",
            TensorField(torch.tensor([seed], device=device)),
        )
        return result.with_field(
            "tracking.trajectory_step",
            TensorField(torch.zeros(1, dtype=torch.long, device=device)),
        )

    @staticmethod
    def _attach_batched_trajectory_identity(
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Attach row identities for one deterministic batched RNG stream."""
        count = len(batch)
        device = batch.latent_origins.device
        seed = -1 if context.seed is None else context.seed
        prefix = "/".join(context.scope.path)
        result = batch.with_field(
            "tracking.trajectory_id",
            ObjectField(
                [f"{prefix}/trajectory[{index}]" for index in range(count)]
            ),
        )
        result = result.with_field(
            "tracking.trajectory_index",
            TensorField(torch.arange(count, device=device)),
        )
        result = result.with_field(
            "tracking.rng_stream_seed",
            TensorField(torch.full((count,), seed, device=device)),
        )
        return result.with_field(
            "tracking.trajectory_step",
            TensorField(torch.zeros(count, dtype=torch.long, device=device)),
        )

    def _run_trajectory(
        self,
        seed: CandidateBatch,
        context: OptimizationContext,
    ) -> list[CandidateBatch]:
        """Run one trajectory while retaining candidates outside its state."""
        # Trajectory point
        current = seed.with_field(
            "local_enumeration.center_sequence",
            ObjectField(seed.sequences),
        )
        emissions: list[CandidateBatch] = []
        elapsed = 0.0
        iteration = 0

        # SORBES trajectory
        while self._should_continue(iteration, elapsed, context):
            iteration_context = context.enter_iteration(iteration)
            current = self.walker(current, iteration_context)
            time_increment = self._time_increment(current)
            if self.include_walk_points:
                emissions.append(current.without_fields(_TRANSIENT_WALKER_FIELDS))
            emissions.append(self._generate_and_filter(current, iteration_context))
            elapsed += time_increment
            iteration += 1
            context.state.record_iteration()
        logger.debug(
            "Local-enumeration trajectory iterations=%s elapsed_walk_time=%.6f "
            "emissions=%s.",
            iteration,
            elapsed,
            len(emissions),
        )
        return emissions

    def _generate_and_filter(
        self,
        point: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Generate MUTANG candidates and apply configured local filters."""
        generated = self.mutation_generator(point, context)
        filtered = self.filters(generated, context)
        return filtered.without_fields(_TRANSIENT_WALKER_FIELDS)

    def _should_continue(
        self,
        iteration: int,
        elapsed: float,
        context: OptimizationContext,
    ) -> bool:
        """Evaluate the configured trajectory bound and global stop state."""
        if context.state.stop_requested:
            return False
        if self.iterations is not None:
            return iteration < self.iterations
        assert self.walk_time is not None
        return elapsed < self.walk_time

    def _batch_should_continue(
        self,
        iteration: int,
        context: OptimizationContext,
    ) -> bool:
        """Return whether a non-empty batch may start another SORBES step."""
        if context.state.stop_requested:
            return False
        return self.iterations is None or iteration < self.iterations

    def _time_increment(self, point: CandidateBatch) -> float:
        """Return adjusted SORBES time or a unit step for fixed iterations."""
        if self.walk_time is None:
            return 1.0
        field = point.fields.get("walker.adjusted_time_step")
        if not isinstance(field, TensorField) or field.values.numel() != 1:
            raise ValueError(
                "Walk-time local enumeration requires one scalar "
                "walker.adjusted_time_step value per trajectory."
            )
        increment = float(field.values.item())
        if increment <= 0:
            raise ValueError("Walker adjusted time step must be positive.")
        return increment

    def _time_increments(self, points: CandidateBatch) -> torch.Tensor:
        """Return one positive adjusted time increment per active trajectory."""
        if self.walk_time is None:
            return torch.ones(
                len(points),
                dtype=points.latent_origins.dtype,
                device=points.latent_origins.device,
            )  # (T_active,)
        field = points.fields.get("walker.adjusted_time_step")
        if not isinstance(field, TensorField):
            raise ValueError(
                "Walk-time local enumeration requires walker.adjusted_time_step."
            )
        increments = field.values.reshape(-1)  # (T_active,)
        if increments.shape[0] != len(points):
            raise ValueError(
                "Walker adjusted time steps must align with active trajectories."
            )
        if not bool(torch.all(torch.isfinite(increments) & (increments > 0)).item()):
            raise ValueError("Walker adjusted time steps must be finite and positive.")
        return increments
