"""Default active/inactive direction sampling from main."""

import torch

from pep_compass.optimization.components.walkers.strategies.sorbes.directions.base import DirectionStrategy
from pep_compass.optimization.components.walkers.strategies.sorbes.schema import SorbesDirections


class ActiveInactiveDirections(DirectionStrategy):
    """Sample unit coordinates in stable and unstable SVD subspaces."""

    def __init__(self, *, vertical_movement: bool = True) -> None:
        self.vertical_movement = vertical_movement

    def __call__(self, geometry):
        active = self._sample(geometry, geometry.active_mask, pull_back=True)
        inactive = (
            self._sample(geometry, ~geometry.active_mask, pull_back=False)
            if self.vertical_movement
            else torch.zeros_like(active)
        )
        return SorbesDirections(active, inactive)

    @staticmethod
    def _sample(geometry, mask, *, pull_back):
        coordinates = torch.randn_like(geometry.singular_values) * mask  # (B, K)
        norm = torch.linalg.vector_norm(coordinates, dim=1, keepdim=True)  # (B, 1)
        coordinates = torch.where(
            norm > 0,
            coordinates / norm.clamp_min(torch.finfo(norm.dtype).eps),
            coordinates,
        )  # (B, K)
        if pull_back:
            coordinates = torch.where(
                mask,
                coordinates / geometry.singular_values.clamp_min(
                    torch.finfo(coordinates.dtype).eps
                ),
                torch.zeros_like(coordinates),
            )  # (B, K)
        return torch.bmm(
            geometry.right_vectors.transpose(1, 2), coordinates.unsqueeze(-1)
        ).squeeze(-1)  # (B, D)
