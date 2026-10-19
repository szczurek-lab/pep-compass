"""Stable top-k score selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.steps.filters.strategies.ranked.selection.base import SelectionRule

if TYPE_CHECKING:
    from pep_compass.optimization.engine.execution.context import OptimizationContext


class TopKSelection(SelectionRule):
    """Keep at most ``k`` best finite-scored candidates."""

    def __init__(self, k: int) -> None:
        """Initialize this object with the declared configuration.

        :param k: K.
        :type k: int
        :rtype: None
        """
        self._configure({"k": k})
        if k < 1:
            raise ValueError("Top-k selection requires a positive k.")
        self.k = k

    def select(
        self,
        scores: torch.Tensor,
        *,
        higher_is_better: bool,
        context: OptimizationContext,
    ) -> torch.Tensor:
        """Select.

        :param scores: Scalar scores or a score tensor aligned with the corresponding candidates.
        :type scores: torch.Tensor
        :param higher_is_better: Higher is better.
        :type higher_is_better: bool
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        del context
        valid = torch.nonzero(~torch.isnan(scores), as_tuple=False).flatten()
        values = scores.index_select(0, valid)
        order = torch.argsort(values, descending=higher_is_better, stable=True)
        return valid.index_select(0, order[: self.k])
