"""Position-update ablation preserved from main/dev."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.steps.walkers.strategies.sorbes.position_update.main import (
    MainPositionUpdate,
)

if TYPE_CHECKING:
    from pep_compass.steps.walkers.strategies.sorbes.schema import SorbesDirections


class WithoutAccelerationPositionUpdate(MainPositionUpdate):
    """Apply the main update after replacing acceleration with zero."""

    def __call__(
        self,
        positions: torch.Tensor,
        directions: SorbesDirections,
        acceleration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Call.

        :param positions: Positions.
        :type positions: torch.Tensor
        :param directions: Direction-selection strategy or sampled numerical directions.
        :type directions: SorbesDirections
        :param acceleration: Acceleration.
        :type acceleration: torch.Tensor
        :return: Result of this operation.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        return super().__call__(positions, directions, torch.zeros_like(acceleration))
