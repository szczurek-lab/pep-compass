"""Article-notation kappa-stable decoder geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.autoencoder.geometry import compute_stable_tangent_geometry
from pep_compass.steps.walkers.strategies.sorbes.geometry.base import GeometryStrategy

if TYPE_CHECKING:
    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.autoencoder.geometry import StableTangentGeometry


class KappaStableGeometry(GeometryStrategy):
    """Compute one reusable decoder SVD and ``S**2 > kappa`` mask."""

    def __init__(self, autoencoder: Autoencoder | None = None, *, kappa: float) -> None:
        """Initialize this object with the declared configuration.

        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder | None
        :param kappa: Kappa.
        :type kappa: float
        :rtype: None
        """
        from pep_compass.utils.validation import validate_positive

        validate_positive(kappa, "KappaStableGeometry.kappa", allow_zero=True)
        self._configure({"kappa": kappa})
        self.autoencoder = autoencoder
        self.kappa = kappa

    def __call__(self, positions: torch.Tensor) -> StableTangentGeometry:
        """Call.

        :param positions: Positions.
        :type positions: torch.Tensor
        :return: Result of this operation.
        :rtype: StableTangentGeometry
        """
        return compute_stable_tangent_geometry(
            self.autoencoder, positions, kappa=self.kappa
        )
