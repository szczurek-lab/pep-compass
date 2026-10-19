"""Boundary-stage contract for SORBES."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pep_compass.utils.configuration import Configurable

if TYPE_CHECKING:
    import torch

    from pep_compass.autoencoder.geometry import StableTangentGeometry


class BoundaryStrategy(Configurable, ABC):
    """Resolve a proposed SORBES point against the supported domain."""

    @abstractmethod
    def __call__(
        self,
        previous: torch.Tensor,
        proposed: torch.Tensor,
        geometry: StableTangentGeometry,
    ) -> torch.Tensor:
        """Return accepted positions.

        :param previous: Previous.
        :type previous: torch.Tensor
        :param proposed: Proposed.
        :type proposed: torch.Tensor
        :param geometry: Geometry strategy or aligned numerical geometry, according to the declared type.
        :type geometry: StableTangentGeometry
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
