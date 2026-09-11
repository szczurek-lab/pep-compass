"""Base contract for score-based candidate selection."""

from abc import ABC, abstractmethod

import torch


class SelectionRule(ABC):
    """Select row indices from scalar candidate scores."""

    @abstractmethod
    def select(self, scores, *, higher_is_better, context) -> torch.Tensor:
        """Return selected indices with shape ``(N,)``."""
