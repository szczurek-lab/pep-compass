"""Geometry supplied by LocalEnumeration."""

from pep_compass.autoencoder.subriemannian import MUTANG_GEOMETRY_REQUIREMENT, PointGeometry
from pep_compass.data.optimization import ObjectField


class SharedGeometry:
    """Consume point-bound geometry without performing a decoder SVD."""

    requirement = MUTANG_GEOMETRY_REQUIREMENT
    local_enumeration_only = True

    def resolve(self, batch, autoencoder=None):
        del autoencoder
        geometries = batch.fields.get("tangent_geometry")
        point_ids = batch.fields.get("point_id")
        if not isinstance(geometries, ObjectField) or not isinstance(point_ids, ObjectField):
            raise ValueError("MUTANG shared geometry requires LocalEnumeration point metadata.")
        for point_id, point_geometry in zip(point_ids.values, geometries.values):
            if not isinstance(point_geometry, PointGeometry) or point_geometry.point_id != point_id:
                raise ValueError("MUTANG received geometry belonging to a different latent point.")
            point_geometry.validate()
        return tuple(value.geometry for value in geometries.values)
