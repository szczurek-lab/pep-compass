"""Default active/inactive direction sampling from main."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.steps.walkers.strategies.sorbes.directions.base import (
    DirectionStrategy,
)
from pep_compass.steps.walkers.strategies.sorbes.schema import SorbesDirections

if TYPE_CHECKING:
    from pep_compass.autoencoder.geometry import StableTangentGeometry


class ActiveInactiveDirections(DirectionStrategy):
    """Sample unit coordinates in stable and unstable SVD subspaces."""

    def __init__(self, *, vertical_movement: bool = True) -> None:
        """Initialize this object with the declared configuration.

        :param vertical_movement: Vertical movement.
        :type vertical_movement: bool
        :rtype: None
        """
        self._configure({"vertical_movement": vertical_movement})
        self.vertical_movement = vertical_movement

    def __call__(self, geometry: StableTangentGeometry) -> SorbesDirections:
        """Call.

        :param geometry: Geometry strategy or aligned numerical geometry, according to the declared type.
        :type geometry: StableTangentGeometry
        :return: Result of this operation.
        :rtype: SorbesDirections
        """
        active = self._sample(geometry, geometry.active_mask, pull_back=True)
        inactive = (
            self._sample(geometry, ~geometry.active_mask, pull_back=False)
            if self.vertical_movement
            else torch.zeros_like(active)
        )
        return SorbesDirections(active, inactive)

    @staticmethod
    def _sample(
        geometry: StableTangentGeometry, mask: torch.Tensor, *, pull_back: torch.Tensor
    ) -> torch.Tensor:
        """Sample.

        :param geometry: Geometry strategy or aligned numerical geometry, according to the declared type.
        :type geometry: StableTangentGeometry
        :param mask: Mask.
        :type mask: torch.Tensor
        :param pull_back: Pull back.
        :type pull_back: torch.Tensor
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
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
                coordinates
                / geometry.singular_values.clamp_min(
                    torch.finfo(coordinates.dtype).eps
                ),
                torch.zeros_like(coordinates),
            )  # (B, K)
        return torch.bmm(
            geometry.right_vectors.transpose(1, 2), coordinates.unsqueeze(-1)
        ).squeeze(-1)  # (B, D)
