"""Stable top-k score selection."""

import torch

from pep_compass.optimization.components.filters.ranked.selection.base import SelectionRule


class TopKSelection(SelectionRule):
    """Keep at most ``k`` best finite-scored candidates."""

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("Top-k selection requires a positive k.")
        self.k = k

    def select(self, scores, *, higher_is_better, context):
        del context
        valid = torch.nonzero(~torch.isnan(scores), as_tuple=False).flatten()
        values = scores.index_select(0, valid)
        order = torch.argsort(values, descending=higher_is_better, stable=True)
        return valid.index_select(0, order[: self.k])
