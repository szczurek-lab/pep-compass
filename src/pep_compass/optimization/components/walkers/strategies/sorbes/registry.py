"""Registries for the fixed, independently replaceable stages of SORBES."""

from pep_compass.optimization.components.walkers.strategies.sorbes.boundary.base import (
    BoundaryStrategy,
)
from pep_compass.optimization.components.walkers.strategies.sorbes.directions.base import (
    DirectionStrategy,
)
from pep_compass.optimization.components.walkers.strategies.sorbes.geometry.base import (
    GeometryStrategy,
)
from pep_compass.optimization.components.walkers.strategies.sorbes.position_update.base import (
    PositionUpdateStrategy,
)
from pep_compass.optimization.components.walkers.strategies.sorbes.scaling.base import (
    ScalingStrategy,
)
from pep_compass.registry import Registry


geometry_registry: Registry[GeometryStrategy] = Registry(
    "SORBES geometry", expected_type=GeometryStrategy
)
directions_registry: Registry[DirectionStrategy] = Registry(
    "SORBES directions", expected_type=DirectionStrategy
)
scaling_registry: Registry[ScalingStrategy] = Registry(
    "SORBES scaling", expected_type=ScalingStrategy
)
position_update_registry: Registry[PositionUpdateStrategy] = Registry(
    "SORBES position update", expected_type=PositionUpdateStrategy
)
boundary_registry: Registry[BoundaryStrategy] = Registry(
    "SORBES boundary", expected_type=BoundaryStrategy
)
