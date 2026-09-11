"""Geometry-stage contract for SORBES."""

from abc import ABC, abstractmethod

import torch

from pep_compass.autoencoder.geometry import StableTangentGeometry


class GeometryStrategy(ABC):
    """Compute geometry attached to one concrete latent point batch."""

    @abstractmethod
    def __call__(self, positions: torch.Tensor) -> StableTangentGeometry:
        """Return geometry for ``positions`` with shape ``(B, D)``."""
