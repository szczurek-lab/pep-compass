"""Direction-scaling contract for SORBES."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pep_compass.utils.configuration import Configurable

if TYPE_CHECKING:
    from pep_compass.autoencoder.geometry import StableTangentGeometry
    from pep_compass.steps.walkers.strategies.sorbes.schema import SorbesDirections


class ScalingStrategy(Configurable, ABC):
    """Scale sampled directions without resampling them."""

    @abstractmethod
    def __call__(
        self, directions: SorbesDirections, geometry: StableTangentGeometry
    ) -> SorbesDirections:
        """Return scaled active and inactive directions.

        :param directions: Direction-selection strategy or sampled numerical directions.
        :type directions: SorbesDirections
        :param geometry: Geometry strategy or aligned numerical geometry, according to the declared type.
        :type geometry: StableTangentGeometry
        :return: Result of this operation.
        :rtype: SorbesDirections
        """
