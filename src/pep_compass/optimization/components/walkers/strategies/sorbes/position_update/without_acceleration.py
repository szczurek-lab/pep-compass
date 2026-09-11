"""Position-update ablation preserved from main/dev."""

import torch

from pep_compass.optimization.components.walkers.strategies.sorbes.position_update.main import MainPositionUpdate


class WithoutAccelerationPositionUpdate(MainPositionUpdate):
    """Apply the main update after replacing acceleration with zero."""

    def __call__(self, positions, directions, acceleration):
        return super().__call__(positions, directions, torch.zeros_like(acceleration))
