"""Mutable run state shared by all scoped execution contexts.

The state records facts produced during execution; it does not schedule steps.
The operation tree decides control flow, while components update counters,
observations, and trust regions through this object. ``Loop`` reads
``stop_requested`` before starting the next iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from pep_compass.data_schemas.execution import OptimizationLimits


class EvaluationObserver(Protocol):
    """Accept completed scoring facts without coupling run state to a database."""

    def observations(
        self,
        objective: str,
        sequences: tuple[str, ...],
        scores: list[float],
        *,
        direction: str,
        lineage: dict[str, list[int]] | None,
        configuration_sha256: str | None = None,
    ) -> None:
        """Retain every completed objective evaluation.

        :param objective: Configured objective field identity.
        :type objective: str
        :param sequences: Newly evaluated sequences in predictor order.
        :type sequences: tuple[str, ...]
        :param scores: Finite scalar results aligned with sequences.
        :type scores: list[float]
        :param direction: Objective ordering, minimize or maximize.
        :type direction: str
        :param lineage: Optional aligned candidate and parent identities.
        :type lineage: dict[str, list[int]] | None
        :param configuration_sha256: Predictor configuration identity, when declared.
        :type configuration_sha256: str | None
        :rtype: None
        """
        ...


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

    REMARK: Parallel branches share this mutable object. The Parallel operation
    orders access to shared services, including in its threaded execution mode.
    """

    limits: OptimizationLimits = field(default_factory=OptimizationLimits)
    observations: dict[str, dict[str, float]] = field(default_factory=dict)
    last_objective: str | None = None
    objective_directions: dict[str, str] = field(default_factory=dict)
    objective_configurations: dict[str, str] = field(default_factory=dict)
    objective_dtypes: dict[str, str] = field(default_factory=dict)
    oracle_calls: int = 0
    generated_candidates: int = 0
    completed_iterations: int = 0
    created_points: int = 0
    created_candidates: int = 0
    stop_requested: bool = False
    trust_regions: dict[str, TrustRegionState] = field(default_factory=dict)
    observer: EvaluationObserver | None = field(default=None, repr=False, compare=False)

    def bind_objective(self, objective: str, configuration_sha256: str) -> None:
        """Bind a field name to one configured predictor for the whole run.

        :param objective: Public objective field name.
        :type objective: str
        :param configuration_sha256: Stable constructor identity.
        :type configuration_sha256: str
        :raises ValueError: If two configurations reuse an objective field.

        :rtype: None
        """
        existing = self.objective_configurations.setdefault(
            objective, configuration_sha256
        )
        if existing != configuration_sha256:
            raise ValueError(f"Conflicting configurations for objective {objective!r}.")

    # Oracle-budget accounting
    def remaining_oracle_calls(self) -> int | None:
        """Return remaining oracle calls, or ``None`` for an unlimited run.

        :return: Result of this operation.
        :rtype: int | None
        """
        if self.limits.oracle_calls is None:
            return None
        return max(self.limits.oracle_calls - self.oracle_calls, 0)

    def record_observations(
        self,
        objective: str,
        sequences: tuple[str, ...],
        scores: list[float],
        *,
        direction: str = "minimize",
        lineage: dict[str, list[int]] | None = None,
    ) -> None:
        """Retain scalar working values and notify the mandatory evaluation recorder.

        :param objective: Stable oracle objective name.
        :param sequences: Evaluated peptide sequences.
        :param scores: Scalar objective values aligned with ``sequences``.

        :type objective: str
        :type sequences: tuple[str, ...]
        :type scores: list[float]
        :param direction: Objective ordering, minimize or maximize.
        :type direction: str
        :param lineage: Candidate-aligned provenance identities.
        :type lineage: dict[str, list[int]] | None
        :rtype: None
        """
        if len(sequences) != len(scores):
            raise ValueError("Observation sequences and scores must have equal length.")
        if direction not in {"minimize", "maximize"}:
            raise ValueError("Objective direction must be minimize or maximize.")
        if not torch.isfinite(torch.as_tensor(scores, dtype=torch.float64)).all():
            raise ValueError("Oracle observations must be finite.")
        if any(len(values) != len(sequences) for values in (lineage or {}).values()):
            raise ValueError("Observation lineage must align with sequences.")
        self.last_objective = objective
        self.objective_directions[objective] = direction
        objective_observations = self.observations.setdefault(objective, {})
        objective_observations.update(zip(sequences, scores))
        self.oracle_calls += len(sequences)
        if (
            self.limits.oracle_calls is not None
            and self.oracle_calls >= self.limits.oracle_calls
        ):
            self.stop_requested = True
        if self.observer is not None:
            self.observer.observations(
                objective,
                sequences,
                scores,
                direction=direction,
                lineage=lineage,
                configuration_sha256=self.objective_configurations.get(objective),
            )

    # Candidate-generation budget accounting
    def record_generated_candidates(self, count: int) -> None:
        """Update generated-candidate count and its optional safety limit.

        :param count: Count.
        :type count: int
        :rtype: None
        """
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

        :type count: int
        :rtype: None
        """
        if count < 0:
            raise ValueError("Completed iteration count cannot be negative.")
        self.completed_iterations += count

    def next_point_ids(self, count: int) -> tuple[int, ...]:
        """Allocate run-local identities for newly computed latent points.

        :param count: Count.
        :type count: int
        :return: Result of this operation.
        :rtype: tuple[int, ...]
        """
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

        :type count: int
        :rtype: tuple[int, ...]
        """
        if count < 0:
            raise ValueError("Candidate identity count cannot be negative.")
        start = self.created_candidates
        self.created_candidates += count
        return tuple(range(start, self.created_candidates))
