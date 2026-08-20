"""Mutable run state shared by all scoped execution contexts.

The state records facts produced during execution; it does not schedule steps.
The operation tree decides control flow, while components update counters,
observations, and trust regions through this object. ``Loop`` reads
``stop_requested`` before starting the next iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class OptimizationLimits:
    """Optional safety limits independent from the configured step tree."""

    oracle_calls: int | None = None
    generated_candidates: int | None = None


@dataclass
class TrustRegionState:
    """Mutable center and radius of one objective-specific trust region."""

    radius: float
    center_sequence: str | None = None
    center_latent: torch.Tensor | None = None
    best_score: float | None = None
    successes: int = 0
    failures: int = 0


@dataclass
class OptimizationState:
    """Store observations, counters, stop state, and trust-region state.

    Update ownership is divided by component role:

    * oracles call :meth:`record_observations`;
    * mutation generators call :meth:`record_generated_candidates`;
    * trust-region filters update ``trust_regions``;
    * ``Loop`` consumes ``stop_requested`` at iteration boundaries.

    REMARK: Parallel branches currently share this mutable object. Concurrent
    updates are therefore not yet guaranteed to be race-free.
    """

    limits: OptimizationLimits = field(default_factory=OptimizationLimits)
    observations: dict[str, dict[str, float]] = field(default_factory=dict)
    observation_latents: dict[str, dict[str, torch.Tensor]] = field(
        default_factory=dict
    )
    oracle_calls: int = 0
    generated_candidates: int = 0
    completed_iterations: int = 0
    created_points: int = 0
    created_candidates: int = 0
    stop_requested: bool = False
    trust_regions: dict[str, TrustRegionState] = field(default_factory=dict)

    # Oracle-budget accounting
    def remaining_oracle_calls(self) -> int | None:
        """Return remaining oracle calls, or ``None`` for an unlimited run."""
        if self.limits.oracle_calls is None:
            return None
        return max(self.limits.oracle_calls - self.oracle_calls, 0)

    def record_observations(
        self,
        objective: str,
        sequences: tuple[str, ...],
        scores: list[float],
        latent_origins: torch.Tensor | None = None,
    ) -> None:
        """Store latest objective values and update the oracle-call counter.

        :param objective: Stable oracle objective name.
        :param sequences: Evaluated peptide sequences.
        :param scores: Scalar objective values aligned with ``sequences``.
        :param latent_origins: Optional aligned latent origins. They are archived
            as detached CPU tensors so completed observations do not retain VRAM.
        """
        if len(sequences) != len(scores):
            raise ValueError("Observation sequences and scores must have equal length.")
        if latent_origins is not None and latent_origins.shape[0] != len(sequences):
            raise ValueError("Observation latent origins must align with sequences.")
        objective_observations = self.observations.setdefault(objective, {})
        objective_observations.update(zip(sequences, scores))
        if latent_origins is not None:
            objective_latents = self.observation_latents.setdefault(objective, {})
            cpu_latents = latent_origins.detach().to(device="cpu", dtype=torch.float32)
            objective_latents.update(
                (sequence, cpu_latents[index].clone())
                for index, sequence in enumerate(sequences)
            )
        self.oracle_calls += len(sequences)
        if (
            self.limits.oracle_calls is not None
            and self.oracle_calls >= self.limits.oracle_calls
        ):
            self.stop_requested = True

    # Candidate-generation budget accounting
    def record_generated_candidates(self, count: int) -> None:
        """Update generated-candidate count and its optional safety limit."""
        self.generated_candidates += count
        if (
            self.limits.generated_candidates is not None
            and self.generated_candidates >= self.limits.generated_candidates
        ):
            self.stop_requested = True

    def record_iteration(self, count: int = 1) -> None:
        """Record completed engine-controlled iterations.

        :param count: Number of logical iterations completed by a batched step.
        :raises ValueError: If ``count`` is negative.
        """
        if count < 0:
            raise ValueError("Completed iteration count cannot be negative.")
        self.completed_iterations += count

    def next_point_ids(self, count: int) -> tuple[int, ...]:
        """Allocate run-local identities for newly computed latent points."""
        start = self.created_points
        self.created_points += count
        return tuple(range(start, self.created_points))

    def next_candidate_ids(self, count: int) -> tuple[int, ...]:
        """Allocate run-local identities for materialized candidates.

        :param count: Number of identities to allocate.
        :return: Monotonically increasing candidate identities.
        :raises ValueError: If ``count`` is negative.

        REMARK: These identifiers describe provenance, not candidate equality.
        Filters may remove rows, while deduplication may collapse equal sequences.
        """
        if count < 0:
            raise ValueError("Candidate identity count cannot be negative.")
        start = self.created_candidates
        self.created_candidates += count
        return tuple(range(start, self.created_candidates))
