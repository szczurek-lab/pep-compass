"""Direct peptide-length constraint."""

import torch

from pep_compass.optimization.components.filters.direct.base import DirectFilter
from pep_compass.optimization.components.filters.manager import FilterManager


@FilterManager.register("sequence_length")
class SequenceLengthConstraint(DirectFilter):
    """Keep sequences whose lengths lie in an inclusive interval."""

    def __init__(self, minimum: int = 1, maximum: int | None = None) -> None:
        if minimum < 0 or (maximum is not None and maximum < minimum):
            raise ValueError("Invalid sequence-length interval.")
        self.minimum, self.maximum = minimum, maximum

    def _execute(self, batch, context):
        del context
        retained = [
            index for index, sequence in enumerate(batch.sequences)
            if len(sequence) >= self.minimum
            and (self.maximum is None or len(sequence) <= self.maximum)
        ]
        return batch.select(torch.tensor(retained, device=batch.latent_origins.device))
