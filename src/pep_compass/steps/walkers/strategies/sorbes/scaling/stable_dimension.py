"""Dimension scaling preserved from main."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pep_compass.steps.walkers.strategies.sorbes.scaling.base import ScalingStrategy
from pep_compass.steps.walkers.strategies.sorbes.schema import SorbesDirections

if TYPE_CHECKING:
    from pep_compass.autoencoder.geometry import StableTangentGeometry


class StableDimensionScaling(ScalingStrategy):
    """Multiply directions by square roots of their subspace dimensions."""

    def __call__(
        self, directions: SorbesDirections, geometry: StableTangentGeometry
    ) -> SorbesDirections:
        """Call.

        :param directions: Direction-selection strategy or sampled numerical directions.
        :type directions: SorbesDirections
        :param geometry: Geometry strategy or aligned numerical geometry, according to the declared type.
        :type geometry: StableTangentGeometry
        :return: Result of this operation.
        :rtype: SorbesDirections
        """
        active_scale = geometry.active_mask.sum(dim=1).sqrt()[:, None]
        inactive_scale = (~geometry.active_mask).sum(dim=1).sqrt()[:, None]
        return SorbesDirections(
            directions.active * active_scale,
            directions.inactive * inactive_scale,
        )
