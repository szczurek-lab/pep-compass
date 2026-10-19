"""Geometry supplied by LocalEnumeration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pep_compass.autoencoder.subriemannian import (
    MUTANG_GEOMETRY_REQUIREMENT,
    PointGeometry,
)
from pep_compass.data_schemas.candidates import ObjectField
from pep_compass.steps.mutations.strategies.mutang.base import (
    GeometryStrategy,
)

if TYPE_CHECKING:
    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.data_schemas.candidates import CandidateBatch


class SharedGeometry(GeometryStrategy):
    """Consume point-bound geometry without performing a decoder SVD."""

    requirement = MUTANG_GEOMETRY_REQUIREMENT
    local_enumeration_only = True

    def resolve(
        self, batch: CandidateBatch, autoencoder: Autoencoder | None = None
    ) -> tuple[object, ...]:
        """Resolve.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder | None
        :return: Result of this operation.
        :rtype: tuple[object, ...]
        """
        del autoencoder
        geometries = batch.fields.get("tangent_geometry")
        point_ids = batch.fields.get("point_id")
        if not isinstance(geometries, ObjectField) or not isinstance(
            point_ids, ObjectField
        ):
            raise ValueError(
                "MUTANG shared geometry requires LocalEnumeration point metadata."
            )
        for point_id, point_geometry in zip(point_ids.values, geometries.values):
            if (
                not isinstance(point_geometry, PointGeometry)
                or point_geometry.point_id != point_id
            ):
                raise ValueError(
                    "MUTANG received geometry belonging to a different latent point."
                )
            point_geometry.validate()
        return tuple(value.geometry for value in geometries.values)
