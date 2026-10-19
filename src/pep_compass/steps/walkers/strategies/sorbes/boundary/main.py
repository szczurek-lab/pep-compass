"""Boundary behaviour preserved from main."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.steps.walkers.strategies.sorbes.boundary.base import BoundaryStrategy

if TYPE_CHECKING:
    from pep_compass.autoencoder.geometry import StableTangentGeometry


class MainBoundary(BoundaryStrategy):
    """Keep the proposal unless the stable subspace is empty."""

    def __call__(
        self,
        previous: torch.Tensor,
        proposed: torch.Tensor,
        geometry: StableTangentGeometry,
    ) -> torch.Tensor:
        # REMARK: main provides no computable W_kappa boundary projection.
        # TODO: Implement the article boundary after W_kappa is fully specified.
        """Call.

        :param previous: Previous.
        :type previous: torch.Tensor
        :param proposed: Proposed.
        :type proposed: torch.Tensor
        :param geometry: Geometry strategy or aligned numerical geometry, according to the declared type.
        :type geometry: StableTangentGeometry
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        valid = geometry.active_mask.any(dim=1)[:, None]
        return torch.where(valid, proposed, previous)
