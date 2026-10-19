"""Historical MUTANG tangent-direction selection."""

from __future__ import annotations

import torch

from pep_compass.steps.mutations.strategies.mutang.base import (
    DirectionStrategy,
)
from pep_compass.steps.mutations.strategies.mutang.schema import (
    SelectedDirections,
)


class BaselineDirectionSelection(DirectionStrategy):
    """Preserve the main/dev significance threshold and minimum count."""

    def __init__(self, threshold: float = 1e-3, minimum: int = 5) -> None:
        """Initialize this object with the declared configuration.

        :param threshold: Threshold.
        :type threshold: float
        :param minimum: Minimum.
        :type minimum: int
        :rtype: None
        """
        from pep_compass.utils.validation import validate_integer, validate_positive

        validate_positive(
            threshold, "BaselineDirectionSelection.threshold", allow_zero=True
        )
        validate_integer(minimum, "BaselineDirectionSelection.minimum")
        self._configure({"threshold": threshold, "minimum": minimum})
        self.threshold, self.minimum = threshold, minimum

    def select(self, singular_values: torch.Tensor) -> SelectedDirections:
        """Select.

        :param singular_values: Singular values.
        :type singular_values: torch.Tensor
        :return: Result of this operation.
        :rtype: SelectedDirections
        """
        significant = int((singular_values > self.threshold).sum().item())
        count = min(max(significant, self.minimum), singular_values.numel())
        return SelectedDirections(torch.arange(count, device=singular_values.device))
