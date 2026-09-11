"""Direction-scaling contract for SORBES."""

from abc import ABC, abstractmethod


class ScalingStrategy(ABC):
    """Scale sampled directions without resampling them."""

    @abstractmethod
    def __call__(self, directions, geometry):
        """Return scaled active and inactive directions."""
