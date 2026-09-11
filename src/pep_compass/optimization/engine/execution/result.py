"""Results returned by the composable optimization runner."""

from dataclasses import dataclass

from pep_compass.data.optimization import Candidate, CandidateBatch


@dataclass(frozen=True)
class OptimizationResult:
    """Final candidates and optional objective summary."""

    candidates: CandidateBatch
    best_candidate: Candidate | None = None
    best_score: float | None = None
    objective_name: str | None = None
    objective_direction: str | None = None
