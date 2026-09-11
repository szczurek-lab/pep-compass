"""Runtime services propagated through the executable step tree.

``PepCompassPipeline.run`` creates the root context. ``Step.__call__`` and the
composite operations derive scoped copies as execution enters a step,
iteration, or branch. All copies intentionally share ``OptimizationState``;
only their path and branch-local random generator differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from pep_compass.optimization.tracking import (
    ExecutionScope,
    NullStepTracker,
    StepTracker,
)
from pep_compass.optimization.engine.execution.state import OptimizationState
from pep_compass.optimization.stability_estimation.monitoring import (
    NullStabilityMonitor,
    StabilityMonitor,
)


@dataclass(frozen=True)
class OptimizationContext:
    """Carry services, scope, RNG, counters, and monitors through execution.

    The context does not decide which component runs next. That order is
    encoded in the ``Step`` tree built by ``PipelineBuilder``. ``Flow``,
    ``Loop``, and ``Parallel`` traverse that tree and pass derived contexts to
    their children.
    """

    autoencoder: Any
    tracker: StepTracker = field(default_factory=NullStepTracker)
    scope: ExecutionScope = field(default_factory=ExecutionScope)
    seed: int | None = None
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    state: OptimizationState = field(default_factory=OptimizationState)
    stability_monitor: StabilityMonitor | NullStabilityMonitor = field(
        default_factory=NullStabilityMonitor
    )

    def enter_step(self, name: str) -> "OptimizationContext":
        """Return a context nested below a named step."""
        return replace(self, scope=self.scope.child(name))

    def enter_iteration(self, index: int) -> "OptimizationContext":
        """Return a context for one loop iteration."""
        return replace(self, scope=self.scope.loop_iteration(index))

    def enter_branch(self, name: str, index: int) -> "OptimizationContext":
        """Return an independently seeded context for a parallel branch."""
        branch_seed = None if self.seed is None else self.seed + index + 1
        return replace(
            self,
            scope=self.scope.branch(name, index),
            seed=branch_seed,
            rng=np.random.default_rng(branch_seed),
        )
