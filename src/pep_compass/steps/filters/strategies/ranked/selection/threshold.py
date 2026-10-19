"""Scalar threshold selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.steps.filters.strategies.ranked.selection.base import SelectionRule

if TYPE_CHECKING:
    from pep_compass.optimization.engine.execution.context import OptimizationContext


class ThresholdSelection(SelectionRule):
    """Keep scores on the accepted side of an inclusive threshold."""

    def __init__(self, threshold: float) -> None:
        """Initialize this object with the declared configuration.

        :param threshold: Threshold.
        :type threshold: float
        :rtype: None
        """
        self._configure({"threshold": threshold})
        self.threshold = threshold

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
        mask = (
            scores >= self.threshold if higher_is_better else scores <= self.threshold
        )
        return torch.nonzero(mask, as_tuple=False).flatten()
