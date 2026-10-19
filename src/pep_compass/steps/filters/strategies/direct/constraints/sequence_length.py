"""Direct peptide-length constraint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.steps.base import Step

if TYPE_CHECKING:
    from pep_compass.data_schemas.candidates import CandidateBatch
    from pep_compass.optimization.engine.execution.context import OptimizationContext


class SequenceLength(Step):
    """Keep sequences whose lengths lie in an inclusive interval."""

    category = "filter"

    def __init__(self, minimum: int = 1, maximum: int | None = None) -> None:
        """Initialize this object with the declared configuration.

        :param minimum: Minimum.
        :type minimum: int
        :param maximum: Maximum.
        :type maximum: int | None
        :rtype: None
        """
        self._configure({"minimum": minimum, "maximum": maximum})
        if minimum < 0 or (maximum is not None and maximum < minimum):
            raise ValueError("Invalid sequence-length interval.")
        self.minimum, self.maximum = minimum, maximum

    def _execute(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> CandidateBatch:
        """Execute.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        del context
        retained = [
            index
            for index, sequence in enumerate(batch.sequences)
            if len(sequence) >= self.minimum
            and (self.maximum is None or len(sequence) <= self.maximum)
        ]
        return batch.select(torch.tensor(retained, device=batch.latent_origins.device))
