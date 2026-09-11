"""Scalar threshold selection."""

import torch

from pep_compass.optimization.components.filters.ranked.selection.base import SelectionRule


class ThresholdSelection(SelectionRule):
    """Keep scores on the accepted side of an inclusive threshold."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def select(self, scores, *, higher_is_better, context):
        del context
        mask = scores >= self.threshold if higher_is_better else scores <= self.threshold
        return torch.nonzero(mask, as_tuple=False).flatten()
