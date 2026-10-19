"""Base contract for score-based candidate selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from pep_compass.utils.configuration import Configurable

if TYPE_CHECKING:
    from pep_compass.optimization.engine.execution.context import OptimizationContext


class SelectionRule(Configurable, ABC):
    """Select row indices from scalar candidate scores."""

    @abstractmethod
    def select(
        self,
        scores: torch.Tensor,
        *,
        higher_is_better: bool,
        context: OptimizationContext,
    ) -> torch.Tensor:
        """Return selected indices with shape ``(N,)``.

        :param scores: Scalar scores or a score tensor aligned with the corresponding candidates.
        :type scores: torch.Tensor
        :param higher_is_better: Higher is better.
        :type higher_is_better: bool
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
