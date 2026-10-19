"""Optimization-engine adapter for SORBES."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.autoencoder.geometry import StableTangentGeometry, TangentDecomposition
from pep_compass.autoencoder.subriemannian import (
    TANGENT_GEOMETRY_CONTRACT,
    PointGeometry,
    SubRiemannianTangentSpace,
)
from pep_compass.data_schemas.candidates import CandidateBatch, ObjectField, TensorField
from pep_compass.steps.base import Step

if TYPE_CHECKING:
    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.optimization.engine.execution.context import OptimizationContext
    from pep_compass.steps.walkers.strategies.sorbes.boundary.base import (
        BoundaryStrategy,
    )
    from pep_compass.steps.walkers.strategies.sorbes.directions.base import (
        DirectionStrategy,
    )
    from pep_compass.steps.walkers.strategies.sorbes.geometry.base import (
        GeometryStrategy,
    )
    from pep_compass.steps.walkers.strategies.sorbes.position_update.base import (
        PositionUpdateStrategy,
    )
    from pep_compass.steps.walkers.strategies.sorbes.scaling.base import ScalingStrategy


class Sorbes(Step):
    """Run SORBES and attach output-point geometry for local enumeration."""

    stage_parameters = (
        "geometry",
        "directions",
        "scaling",
        "position_update",
        "boundary",
    )
    category = "walker"

    required_resources = ("autoencoder",)
    output_fields = (
        "tangent_geometry",
        "point_id",
        "walker.tangent_space",
        "walker.singular_values",
        "walker.left_vectors",
        "walker.adjusted_time_step",
    )
    invalidated_prefixes = ("oracle.",)
    display_parameters = ("geometry", "position_update", "directions")

    def __init__(
        self,
        *,
        geometry: GeometryStrategy | None = None,
        directions: DirectionStrategy | None = None,
        scaling: ScalingStrategy | None = None,
        position_update: PositionUpdateStrategy | None = None,
        boundary: BoundaryStrategy | None = None,
        time_step: float = 0.01,
        horizontal_threshold: float = 0.1,
        max_horizontal_update_norm: float = 0.5,
        vertical_movement: bool = True,
    ) -> None:
        """Configure SORBES without loading resources.

        :param geometry: Decoder geometry strategy; defaults to kappa=threshold squared.
        :param time_step: Positive diffusion step used by the default position update.
        :param horizontal_threshold: Non-negative stable singular-value threshold.
        :param max_horizontal_update_norm: Positive norm bound for the default update.
        :param vertical_movement: Include inactive decoder directions.

        :type geometry: GeometryStrategy | None
        :param directions: Direction-selection strategy or sampled numerical directions.
        :type directions: DirectionStrategy | None
        :param scaling: Strategy scaling sampled directions.
        :type scaling: ScalingStrategy | None
        :param position_update: Strategy computing the next bounded latent position.
        :type position_update: PositionUpdateStrategy | None
        :param boundary: Strategy accepting or modifying a proposed position.
        :type boundary: BoundaryStrategy | None
        :type time_step: float
        :type horizontal_threshold: float
        :type max_horizontal_update_norm: float
        :type vertical_movement: bool
        :rtype: None
        """
        from math import sqrt

        from pep_compass.utils.validation import validate_positive

        from .boundary.main import MainBoundary
        from .directions.active_inactive import ActiveInactiveDirections
        from .geometry.kappa_stable import KappaStableGeometry
        from .position_update.main import MainPositionUpdate
        from .scaling.stable_dimension import StableDimensionScaling

        validate_positive(time_step, "Sorbes.time_step")
        validate_positive(
            max_horizontal_update_norm, "Sorbes.max_horizontal_update_norm"
        )
        if (
            isinstance(horizontal_threshold, bool)
            or not isinstance(horizontal_threshold, (int, float))
            or horizontal_threshold < 0
        ):
            raise ValueError("Sorbes.horizontal_threshold must be non-negative.")
        if not isinstance(vertical_movement, bool):
            raise TypeError("vertical_movement must be boolean.")
        self.geometry = (
            geometry
            if geometry is not None
            else KappaStableGeometry(kappa=horizontal_threshold**2)
        )
        self.directions = (
            directions
            if directions is not None
            else ActiveInactiveDirections(vertical_movement=vertical_movement)
        )
        self.scaling = scaling if scaling is not None else StableDimensionScaling()
        self.position_update = (
            position_update
            if position_update is not None
            else MainPositionUpdate(
                epsilon=sqrt(time_step), delta_max=max_horizontal_update_norm
            )
        )
        self.boundary = boundary if boundary is not None else MainBoundary()
        self._configure(
            dict(
                geometry=self.geometry,
                directions=self.directions,
                scaling=self.scaling,
                position_update=self.position_update,
                boundary=self.boundary,
                time_step=time_step,
                horizontal_threshold=horizontal_threshold,
                max_horizontal_update_norm=max_horizontal_update_norm,
                vertical_movement=vertical_movement,
            )
        )
        self.geometry_contract = TANGENT_GEOMETRY_CONTRACT

    def _initialize(self, autoencoder: Autoencoder | None) -> None:
        """Initialize.

        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder | None
        :rtype: None
        """
        from .algorithm import SorbesAlgorithm

        self.sorbes = SorbesAlgorithm(
            autoencoder,
            self.geometry,
            self.directions,
            self.scaling,
            self.position_update,
            self.boundary,
        )

    def _execute(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> CandidateBatch:
        """Execute.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        current = self._current_geometry(batch)
        step = self.sorbes.step(batch.latent_origins, current, context=context)
        sequences = context.autoencoder.decode_peptides(step.positions)
        fields = {
            name: value
            for name, value in batch.fields.items()
            if not name.startswith("oracle.")
        }
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
            PointGeometry(
                point_ids[index], step.geometry.select(index), id(self.sorbes.geometry)
            )
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
    def _current_geometry(batch: CandidateBatch) -> StableTangentGeometry | None:
        """Current geometry.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :return: Result of this operation.
        :rtype: StableTangentGeometry | None
        """
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
