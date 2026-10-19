"""Minimal components used to validate engine composition contracts."""

from pep_compass.data_schemas.candidates import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.steps.base import Step


class SuffixStep(Step):
    """Append a configured suffix to every candidate sequence."""

    def __init__(self, suffix: str) -> None:
        self._configure({"suffix": suffix})
        self.suffix = suffix

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Append the suffix while retaining all aligned fields."""
        return batch.with_sequences(
            [f"{sequence}{self.suffix}" for sequence in batch.sequences]
        )
