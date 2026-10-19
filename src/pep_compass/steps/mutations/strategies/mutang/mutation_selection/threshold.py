"""Baseline threshold selection of single mutations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pep_compass.steps.mutations.strategies.mutang.base import (
    MutationSelectionStrategy,
)
from pep_compass.steps.mutations.strategies.mutang.schema import (
    MutationOption,
)

if TYPE_CHECKING:
    import torch


class ThresholdSelection(MutationSelectionStrategy):
    """Select every non-padding residue above the token threshold."""

    def __init__(self, threshold: float = 0.1) -> None:
        """Initialize this object with the declared configuration.

        :param threshold: Threshold.
        :type threshold: float
        :rtype: None
        """
        from pep_compass.utils.validation import validate_positive

        validate_positive(threshold, "ThresholdSelection.threshold", allow_zero=True)
        self._configure({"threshold": threshold})
        self.threshold = threshold

    def select(
        self, scores: torch.Tensor, sequence_length: int
    ) -> tuple[MutationOption, ...]:
        """Select.

        :param scores: Scalar scores or a score tensor aligned with the corresponding candidates.
        :type scores: torch.Tensor
        :param sequence_length: Sequence length.
        :type sequence_length: int
        :return: Result of this operation.
        :rtype: tuple[MutationOption, ...]
        """
        rows = (scores[:sequence_length, 1:] > self.threshold).nonzero(as_tuple=False)
        return tuple(
            MutationOption(
                int(position), int(residue) + 1, float(scores[position, residue + 1])
            )
            for position, residue in rows.cpu().tolist()
        )
