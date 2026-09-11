"""Execution lifecycle shared by every operation and concrete component.

The core builder creates a tree of :class:`Step` objects. At runtime,
``PepCompassPipeline.run`` creates one :class:`OptimizationContext` and invokes
the root step. The classes in this package define how that invocation is
wrapped, scoped, measured, and stopped; they do not select the configured
component order.
"""

from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.result import OptimizationResult
from pep_compass.optimization.engine.execution.state import (
    OptimizationLimits,
    OptimizationState,
)
from pep_compass.optimization.engine.execution.step import Step

__all__ = [
    "OptimizationContext",
    "OptimizationLimits",
    "OptimizationResult",
    "OptimizationState",
    "Step",
]
