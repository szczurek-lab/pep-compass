"""Runtime services propagated through the executable step tree.

``ExecutablePipeline.run`` creates the root context. ``Step.__call__`` and the
composite operations derive scoped copies as execution enters a step,
iteration, or branch. All copies intentionally share ``OptimizationState``;
only their path and branch-local random generator differ.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from pep_compass.optimization.engine.execution.state import OptimizationState
from pep_compass.optimization.stability_estimation.monitoring import (
    NullStabilityMonitor,
    StabilityMonitor,
)
from pep_compass.optimization.tracking import (
    ExecutionScope,
    NullStepTracker,
    StepTracker,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.optimization.saving.saver import Saver


@dataclass(frozen=True)
class OptimizationContext:
    """Carry services, scope, RNG, counters, and monitors through execution.

    The context does not decide which component runs next. That order is
    encoded in the ``Step`` tree built by ``PipelineBuilder``. ``Flow``,
    ``Loop``, and ``Parallel`` traverse that tree and pass derived contexts to
    their children.
    """

    autoencoder: Autoencoder
    tracker: StepTracker = field(default_factory=NullStepTracker)
    scope: ExecutionScope = field(default_factory=ExecutionScope)
    seed: int | None = None
    rng: np.random.Generator | None = None
    state: OptimizationState = field(default_factory=OptimizationState)
    stability_monitor: StabilityMonitor | NullStabilityMonitor = field(
        default_factory=NullStabilityMonitor
    )
    saver: Saver | None = None
    node_id: str | None = None
    step_state: dict[str, object] = field(default_factory=dict)

    def enter_node(self, node_id: str | None) -> "OptimizationContext":
        """Return a scope bound to a stable declaration identity.

        :param node_id: Stable graph placement identity.
        :type node_id: str | None
        :return: Result of this operation.
        :rtype: OptimizationContext
        """
        return replace(self, node_id=node_id)

    def measure_stage(
        self,
        name: str,
        *,
        input_count: int | None = None,
        output_count: int | None = None,
    ) -> AbstractContextManager[None]:
        """Measure an internal stage through the current run's common recorder.

        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :param input_count: Input count.
        :type input_count: int | None
        :param output_count: Output count.
        :type output_count: int | None
        :return: Result of this operation.
        :rtype: AbstractContextManager[None]
        """
        from contextlib import nullcontext

        return (
            self.saver.stage(name, input_count=input_count, output_count=output_count)
            if self.saver is not None
            else nullcontext()
        )

    def iteration_completed(self, index: int, *, trajectories: int = 1) -> None:
        """Emit one operator iteration independently of trajectory counters.

        :param index: Index.
        :type index: int
        :param trajectories: Independent trajectories per input seed.
        :type trajectories: int
        :rtype: None
        """
        if self.saver is not None:
            self.saver.iteration_completed(self, index, trajectories)

    def iteration_started(self, index: int) -> None:
        """Mark one operator iteration without changing computational counters.

        :param index: Index.
        :type index: int
        :rtype: None
        """
        if self.saver is not None:
            self.saver.iteration_started(self, index)

    def __post_init__(self) -> None:
        """Create a seeded local stream when callers do not provide one.

        :rtype: None
        """
        if self.rng is None:
            object.__setattr__(self, "rng", np.random.default_rng(self.seed))

    def enter_step(self, name: str) -> "OptimizationContext":
        """Return a context nested below a named step.

        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :return: Result of this operation.
        :rtype: OptimizationContext
        """
        return replace(self, scope=self.scope.child(name))

    def enter_iteration(self, index: int) -> "OptimizationContext":
        """Return a context for one loop iteration.

        :param index: Index.
        :type index: int
        :return: Result of this operation.
        :rtype: OptimizationContext
        """
        return replace(self, scope=self.scope.loop_iteration(index))

    def enter_branch(self, name: str, index: int) -> "OptimizationContext":
        """Return an independently seeded context for a parallel branch.

        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :param index: Index.
        :type index: int
        :return: Result of this operation.
        :rtype: OptimizationContext
        """
        # A consumed parent-stream token distinguishes repeated branch calls;
        # structural scope separates nested branches and loop iterations.
        token = int(self.rng.integers(0, 2**63))
        payload = json.dumps(
            (
                self.seed,
                self.scope.path,
                self.scope.loop_indices,
                self.scope.branch_names,
                name,
                index,
                token,
            )
        ).encode()
        branch_seed = (
            None
            if self.seed is None
            else int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)
        )
        return replace(
            self,
            scope=self.scope.branch(name, index),
            seed=branch_seed,
            rng=np.random.default_rng(branch_seed),
        )
