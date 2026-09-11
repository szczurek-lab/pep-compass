"""Dimension scaling preserved from main."""

from pep_compass.optimization.components.walkers.strategies.sorbes.scaling.base import ScalingStrategy
from pep_compass.optimization.components.walkers.strategies.sorbes.schema import SorbesDirections


class StableDimensionScaling(ScalingStrategy):
    """Multiply directions by square roots of their subspace dimensions."""

    def __call__(self, directions, geometry):
        active_scale = geometry.active_mask.sum(dim=1).sqrt()[:, None]
        inactive_scale = (~geometry.active_mask).sum(dim=1).sqrt()[:, None]
        return SorbesDirections(
            directions.active * active_scale,
            directions.inactive * inactive_scale,
        )
