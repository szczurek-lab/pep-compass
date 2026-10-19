"""Standalone MUTANG geometry computed explicitly from an autoencoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pep_compass.autoencoder.geometry import compute_stable_tangent_geometry
from pep_compass.steps.mutations.strategies.mutang.base import (
    GeometryStrategy,
)

if TYPE_CHECKING:
    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.data_schemas.candidates import CandidateBatch


class KappaStableGeometry(GeometryStrategy):
    """Compute independent point geometries for standalone MUTANG execution."""

    local_enumeration_only = False

    def __init__(self, *, kappa: float = 1e-6) -> None:
        """Initialize this object with the declared configuration.

        :param kappa: Kappa.
        :type kappa: float
        :rtype: None
        """
        self._configure({"kappa": kappa})
        self.kappa = kappa

    def resolve(
        self, batch: CandidateBatch, autoencoder: Autoencoder | None
    ) -> tuple[object, ...]:
        """Resolve.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder | None
        :return: Result of this operation.
        :rtype: tuple[object, ...]
        """
        geometry = compute_stable_tangent_geometry(
            autoencoder, batch.latent_origins, kappa=self.kappa
        )
        return tuple(geometry.select(index) for index in range(len(batch)))
