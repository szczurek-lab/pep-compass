"""Repeated traversal of one configured child flow.

``PipelineBuilder`` wraps YAML ``loop.steps`` in a ``Flow`` and installs it as
the loop body. Each iteration receives the previous iteration's output and a
scope containing the iteration index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pep_compass.data_schemas.candidates import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.steps.base import Step
from pep_compass.utils.logger import get_custom_logger

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = get_custom_logger(__name__)


class Loop(Step):
    """Execute one child step a fixed number of times."""

    display_parameters = ("iterations",)
    category = "flow"
    kind = "loop"

    def __init__(self, *, steps: Sequence[Step], iterations: int) -> None:
        """Initialize this object with the declared configuration.

        :param steps: Ordered child step instances.
        :type steps: Sequence[Step]
        :param iterations: Number of complete logical iterations.
        :type iterations: int
        :rtype: None
        """
        from pep_compass.steps.flow.strategies.sequence import Flow
        from pep_compass.utils.validation import validate_integer

        validate_integer(iterations, "Loop.iterations")
        self.body = Flow(steps)
        self.iterations = iterations
        self._configure(dict(iterations=iterations))

    def graph_children(
        self,
    ) -> dict[str, Step | tuple[Step, ...] | dict[str, tuple[Step, ...]]]:
        """Graph children.

        :return: Result of this operation.
        :rtype: dict[str, Step | tuple[Step, ...] | dict[str, tuple[Step, ...]]]
        """
        return {"steps": self.body.steps}

    def precompute(self, context: OptimizationContext) -> None:
        """Precompute the repeated body once.

        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :rtype: None
        """
        self.body.precompute(context.enter_step(self.body.name))

    def _execute(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> CandidateBatch:
        """Execute iterations until completion or a runtime stop condition.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        result = batch
        for index in range(self.iterations):
            if context.state.stop_requested or len(result) == 0:
                break
            previous_result = result
            context.iteration_started(index)
            iteration_result = self.body(result, context.enter_iteration(index))
            context.state.record_iteration()
            context.iteration_completed(index)

            # Iteration commit
            ## Preserve the last valid result when a component uses an empty
            ## batch as an explicit search-exhaustion signal.
            ### REMARK: Ordinary empty filter results remain terminal. Rollback
            ### applies only when the component also requests the run to stop.
            if len(iteration_result) == 0 and context.state.stop_requested:
                logger.info(
                    "Loop stopped at iteration=%s after candidate exhaustion; "
                    "preserving previous_candidates=%s.",
                    index,
                    len(previous_result),
                )
                result = previous_result
                break
            result = iteration_result
        return result
