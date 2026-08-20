"""Built-in walker strategies."""

import math

from pep_compass.optimization.components.walkers.strategies.sorbes import (
    Sorbes,
    SorbesStrategyManager,
    SorbesWalker,
)
from pep_compass.optimization.components.walkers.manager import WalkerManager


@WalkerManager.register("sorbes")
def build_sorbes(
    autoencoder,
    *,
    geometry=None,
    directions=None,
    scaling=None,
    position_update=None,
    boundary=None,
    horizontal_threshold=None,
    time_step=None,
    max_horizontal_update_norm=None,
    vertical_movement=True,
):
    """Build the fixed SORBES algorithm from nested stage declarations."""
    # SORBES retains a fixed stage order while every stage implementation is
    # selected from its dedicated registry.
    if geometry is None:
        geometry = {
            "method": "kappa_stable",
            "parameters": {"kappa": float(horizontal_threshold or 0.1) ** 2},
        }
    if directions is None:
        directions = {
            "method": "active_inactive",
            "parameters": {"vertical_movement": vertical_movement},
        }
    if position_update is None:
        position_update = {
            "method": "main",
            "parameters": {
                "epsilon": math.sqrt(float(time_step or 0.01)),
                "delta_max": float(max_horizontal_update_norm or 0.5),
            },
        }
    if scaling is None:
        scaling = {"method": "stable_dimension", "parameters": {}}
    if boundary is None:
        boundary = {"method": "main", "parameters": {}}
    stages = SorbesStrategyManager.build(
        autoencoder,
        {
            "geometry": geometry,
            "directions": directions,
            "scaling": scaling,
            "position_update": position_update,
            "boundary": boundary,
        },
    )
    return SorbesWalker(Sorbes(autoencoder, *stages))

__all__ = [
    "Sorbes",
    "SorbesWalker",
]
