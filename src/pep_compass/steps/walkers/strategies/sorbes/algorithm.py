"""Fixed SORBES algorithm assembled from replaceable numerical stages."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from pep_compass.steps.walkers.strategies.sorbes.schema import SorbesStepResult

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.autoencoder.geometry import StableTangentGeometry
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


class SorbesAlgorithm:
    """Advance points in the only supported SORBES stage order."""

    def __init__(
        self,
        autoencoder: Autoencoder,
        geometry: GeometryStrategy,
        directions: DirectionStrategy,
        scaling: ScalingStrategy,
        position_update: PositionUpdateStrategy,
        boundary: BoundaryStrategy,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder
        :param geometry: Geometry strategy or aligned numerical geometry, according to the declared type.
        :type geometry: GeometryStrategy
        :param directions: Direction-selection strategy or sampled numerical directions.
        :type directions: DirectionStrategy
        :param scaling: Strategy scaling sampled directions.
        :type scaling: ScalingStrategy
        :param position_update: Strategy computing the next bounded latent position.
        :type position_update: PositionUpdateStrategy
        :param boundary: Strategy accepting or modifying a proposed position.
        :type boundary: BoundaryStrategy
        :rtype: None
        """
        self.autoencoder = autoencoder
        self.geometry = geometry
        self.directions = directions
        self.scaling = scaling
        self.position_update = position_update
        self.boundary = boundary

    def step(
        self,
        positions: torch.Tensor,
        current_geometry: StableTangentGeometry | None = None,
        *,
        context: OptimizationContext | None = None,
    ) -> SorbesStepResult:
        """Move points and return geometry belonging to the resulting points.

        :param positions: Positions.
        :type positions: torch.Tensor
        :param current_geometry: Current geometry.
        :type current_geometry: StableTangentGeometry | None
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext | None
        :return: Result of this operation.
        :rtype: SorbesStepResult
        """

        def measure(name: str) -> AbstractContextManager[None]:
            """Measure.

            :param name: Name identifying this object or recorded artifact.
            :type name: str
            :return: Result of this operation.
            :rtype: AbstractContextManager[None]
            """
            return (
                context.measure_stage(
                    name, input_count=len(positions), output_count=len(positions)
                )
                if context is not None
                else nullcontext()
            )

        # Fixed numerical stage order; measurement does not alter RNG or tensor operations.
        with measure("input_geometry"):
            geometry = current_geometry or self.geometry(positions)
        with measure("directions"):
            raw_directions = self.directions(geometry)
        with measure("scaling"):
            directions = self.scaling(raw_directions, geometry)
        with measure("acceleration"):
            with torch.no_grad():
                ambient = self.autoencoder.field_derivative(
                    positions, directions.active
                )  # (B, A)
            acceleration = geometry.project_ambient_to_active_latent(ambient)  # (B, D)
        with measure("position_update"):
            proposed, time_steps = self.position_update(
                positions, directions, acceleration
            )  # (B, D), (B,)
        with measure("boundary"):
            accepted = self.boundary(positions, proposed, geometry)  # (B, D)
        with measure("output_geometry"):
            output_geometry = self.geometry(accepted)
        return SorbesStepResult(accepted, time_steps, output_geometry)
