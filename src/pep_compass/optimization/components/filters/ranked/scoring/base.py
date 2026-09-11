"""Base contract for reusable candidate scoring."""

from abc import ABC, abstractmethod

import torch


class ScoreFunction(ABC):
    """Assign one scalar score to every candidate in a batch."""

    higher_is_better: bool = True

    @abstractmethod
    def __call__(self, batch, context) -> torch.Tensor:
        """Return scores with shape ``(B,)``."""
