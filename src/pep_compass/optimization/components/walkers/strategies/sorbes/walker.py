"""Optimization-engine adapter for SORBES."""

import torch

from pep_compass.autoencoder.geometry import StableTangentGeometry, TangentDecomposition
from pep_compass.autoencoder.subriemannian import PointGeometry, SubRiemannianTangentSpace, TANGENT_GEOMETRY_CONTRACT
from pep_compass.data.optimization import CandidateBatch, ObjectField, TensorField
from pep_compass.optimization.components.walkers.base import Walker


class SorbesWalker(Walker):
    """Run SORBES and attach output-point geometry for local enumeration."""

    def __init__(self, sorbes) -> None:
        self.sorbes = sorbes
        self.geometry_contract = TANGENT_GEOMETRY_CONTRACT

    def _execute(self, batch, context):
        current = self._current_geometry(batch)
        step = self.sorbes.step(batch.latent_origins, current)
        sequences = context.autoencoder.decode_peptides(step.positions)
        fields = dict(batch.fields)
        point_ids = context.state.next_point_ids(len(batch))
        parent_ids = fields.get("lineage.candidate_id")
        if isinstance(parent_ids, TensorField):
            fields["lineage.parent_candidate_id"] = TensorField(parent_ids.values)
        fields["lineage.candidate_id"] = TensorField(
            torch.as_tensor(
                context.state.next_candidate_ids(len(batch)),
                dtype=torch.long,
                device=step.positions.device,
            )
        )
        point_geometries = [
            PointGeometry(point_ids[index], step.geometry.select(index), id(self.sorbes.geometry))
            for index in range(len(batch))
        ]
        fields["tangent_geometry"] = ObjectField(point_geometries)
        fields["point_id"] = ObjectField(point_ids)
        fields["walker.singular_values"] = TensorField(step.geometry.singular_values)
        fields["walker.left_vectors"] = TensorField(step.geometry.left_vectors)
        fields["walker.adjusted_time_step"] = TensorField(step.adjusted_time_steps)
        threshold = step.geometry.kappa**0.5
        fields["walker.tangent_space"] = ObjectField(
            [
                SubRiemannianTangentSpace(
                    step.geometry.left_vectors[index],
                    step.geometry.singular_values[index],
                    step.geometry.right_vectors[index],
                    threshold,
                )
                for index in range(len(batch))
            ]
        )
        return CandidateBatch(sequences, step.positions, fields)

    @staticmethod
    def _current_geometry(batch):
        field = batch.fields.get("tangent_geometry")
        if not isinstance(field, ObjectField) or not field.values:
            return None
        if not all(isinstance(value, PointGeometry) for value in field.values):
            return None
        values = [value.geometry for value in field.values]
        decomposition = TangentDecomposition(
            torch.cat([value.left_vectors for value in values]),
            torch.cat([value.singular_values for value in values]),
            torch.cat([value.right_vectors for value in values]),
        )
        return StableTangentGeometry(
            decomposition,
            values[0].kappa,
            torch.cat([value.active_mask for value in values]),
        )
