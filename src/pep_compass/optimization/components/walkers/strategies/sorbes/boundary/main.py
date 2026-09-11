"""Boundary behaviour preserved from main."""

import torch

from pep_compass.optimization.components.walkers.strategies.sorbes.boundary.base import BoundaryStrategy


class MainBoundary(BoundaryStrategy):
    """Keep the proposal unless the stable subspace is empty."""

    def __call__(self, previous, proposed, geometry):
        # REMARK: main provides no computable W_kappa boundary projection.
        # TODO: Implement the article boundary after W_kappa is fully specified.
        valid = geometry.active_mask.any(dim=1)[:, None]
        return torch.where(valid, proposed, previous)
