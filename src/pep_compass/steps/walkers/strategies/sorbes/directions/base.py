"""Direction-stage contract for SORBES."""

from abc import ABC, abstractmethod

from pep_compass.autoencoder.geometry import StableTangentGeometry
from pep_compass.steps.walkers.strategies.sorbes.schema import SorbesDirections
from pep_compass.utils.configuration import Configurable


class DirectionStrategy(Configurable, ABC):
    """Generate active and inactive directions from fixed point geometry."""

    @abstractmethod
    def __call__(self, geometry: StableTangentGeometry) -> SorbesDirections:
        """Return directions without modifying the shared geometry.

        :param geometry: Geometry strategy or aligned numerical geometry, according to the declared type.
        :type geometry: StableTangentGeometry
        :return: Result of this operation.
        :rtype: SorbesDirections
        """
