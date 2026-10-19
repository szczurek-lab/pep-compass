"""Geometry-stage contract for SORBES."""

from abc import ABC, abstractmethod

import torch

from pep_compass.autoencoder.geometry import StableTangentGeometry
from pep_compass.utils.configuration import Configurable


class GeometryStrategy(Configurable, ABC):
    """Compute geometry attached to one concrete latent point batch."""

    @abstractmethod
    def __call__(self, positions: torch.Tensor) -> StableTangentGeometry:
        """Return geometry for ``positions`` with shape ``(B, D)``.

        :param positions: Positions.
        :type positions: torch.Tensor
        :return: Result of this operation.
        :rtype: StableTangentGeometry
        """
