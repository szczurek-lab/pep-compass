"""Construction of the fixed SORBES stage set."""

from pep_compass.optimization.components.walkers.strategies.sorbes.boundary.main import MainBoundary
from pep_compass.optimization.components.walkers.strategies.sorbes.directions.active_inactive import ActiveInactiveDirections
from pep_compass.optimization.components.walkers.strategies.sorbes.geometry.kappa_stable import KappaStableGeometry
from pep_compass.optimization.components.walkers.strategies.sorbes.position_update.article import ArticlePositionUpdate
from pep_compass.optimization.components.walkers.strategies.sorbes.position_update.main import MainPositionUpdate
from pep_compass.optimization.components.walkers.strategies.sorbes.position_update.without_acceleration import WithoutAccelerationPositionUpdate
from pep_compass.optimization.components.walkers.strategies.sorbes.scaling.stable_dimension import StableDimensionScaling
from pep_compass.optimization.components.walkers.strategies.sorbes.registry import (
    boundary_registry,
    directions_registry,
    geometry_registry,
    position_update_registry,
    scaling_registry,
)


@geometry_registry.register("kappa_stable", services={"autoencoder"})
def build_kappa_stable_geometry(autoencoder, *, kappa: float) -> KappaStableGeometry:
    """Build the standard stable-tangent geometry stage."""
    return KappaStableGeometry(autoencoder, kappa=kappa)


@directions_registry.register("active_inactive")
def build_active_inactive_directions(
    *, vertical_movement: bool = True
) -> ActiveInactiveDirections:
    """Build the standard active/inactive direction sampler."""
    return ActiveInactiveDirections(vertical_movement=vertical_movement)


@scaling_registry.register("stable_dimension")
def build_stable_dimension_scaling() -> StableDimensionScaling:
    """Build the standard stable-dimension scaling stage."""
    return StableDimensionScaling()


@position_update_registry.register("main")
def build_main_position_update(*, epsilon: float, delta_max: float) -> MainPositionUpdate:
    """Build the reference accelerated position update."""
    return MainPositionUpdate(epsilon=epsilon, delta_max=delta_max)


@position_update_registry.register("article")
def build_article_position_update(*, epsilon: float, delta_max: float) -> ArticlePositionUpdate:
    """Build the article-form position update."""
    return ArticlePositionUpdate(epsilon=epsilon, delta_max=delta_max)


@position_update_registry.register("without_acceleration")
def build_without_acceleration_position_update(
    *, epsilon: float, delta_max: float
) -> WithoutAccelerationPositionUpdate:
    """Build the no-acceleration position update."""
    return WithoutAccelerationPositionUpdate(epsilon=epsilon, delta_max=delta_max)


@boundary_registry.register("main")
def build_main_boundary() -> MainBoundary:
    """Build the standard SORBES domain-boundary stage."""
    return MainBoundary()


class SorbesStrategyManager:
    """Build supported stage implementations without changing stage order."""

    @classmethod
    def build(cls, autoencoder, parameters):
        """Construct all stages from explicit nested declarations."""
        geometry = parameters["geometry"]
        directions = parameters["directions"]
        scaling = parameters["scaling"]
        position_update = parameters["position_update"]
        boundary = parameters["boundary"]
        return (
            geometry_registry.build(
                geometry["method"], geometry.get("parameters", {}),
                services={"autoencoder": autoencoder},
            ),
            directions_registry.build(
                directions["method"], directions.get("parameters", {}),
            ),
            scaling_registry.build(
                scaling["method"], scaling.get("parameters", {}),
            ),
            position_update_registry.build(
                position_update["method"], position_update.get("parameters", {}),
            ),
            boundary_registry.build(
                boundary["method"], boundary.get("parameters", {}),
            ),
        )
