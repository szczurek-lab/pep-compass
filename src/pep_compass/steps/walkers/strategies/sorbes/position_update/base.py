"""Position-update contract for SORBES."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pep_compass.utils.configuration import Configurable

if TYPE_CHECKING:
    import torch

    from pep_compass.steps.walkers.strategies.sorbes.schema import SorbesDirections


class PositionUpdateStrategy(Configurable, ABC):
    """Calculate the bounded displacement from one SORBES point."""

    @abstractmethod
    def __call__(
        self,
        positions: torch.Tensor,
        directions: SorbesDirections,
        acceleration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(new_positions, effective_time_steps)``.

        :param positions: Positions.
        :type positions: torch.Tensor
        :param directions: Direction-selection strategy or sampled numerical directions.
        :type directions: SorbesDirections
        :param acceleration: Acceleration.
        :type acceleration: torch.Tensor
        :return: Result of this operation.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
