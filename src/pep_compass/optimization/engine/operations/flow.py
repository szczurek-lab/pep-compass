"""Sequential traversal of configured child steps.

``PipelineBuilder`` creates a ``Flow`` for every YAML ``steps`` list. During
execution it passes each returned ``CandidateBatch`` directly to the next
child. Every child enters the standard ``Step.__call__`` lifecycle.
"""

from collections.abc import Sequence

from pep_compass.data.optimization import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.step import Step


class Flow(Step):
    """Execute child steps sequentially in their configured order."""

    def __init__(self, steps: Sequence[Step]) -> None:
        self.steps = tuple(steps)

    def precompute(self, context: OptimizationContext) -> None:
        """Precompute every child in execution order."""
        for step in self.steps:
            step.precompute(context.enter_step(step.name))

    def _execute(self, batch: CandidateBatch, context: OptimizationContext) -> CandidateBatch:
        """Pass each intermediate batch to the next child step."""
        result = batch
        for step in self.steps:
            result = step(result, context)
        return result
