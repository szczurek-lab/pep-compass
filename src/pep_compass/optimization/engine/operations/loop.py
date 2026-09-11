"""Repeated traversal of one configured child flow.

``PipelineBuilder`` wraps YAML ``loop.steps`` in a ``Flow`` and installs it as
the loop body. Each iteration receives the previous iteration's output and a
scope containing the iteration index.
"""

from pep_compass.data.optimization import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.step import Step
from pep_compass.utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


class Loop(Step):
    """Execute one child step a fixed number of times."""

    def __init__(self, body: Step, iterations: int) -> None:
        if iterations < 0:
            raise ValueError("Loop iterations cannot be negative.")
        self.body = body
        self.iterations = iterations

    def precompute(self, context: OptimizationContext) -> None:
        """Precompute the repeated body once."""
        self.body.precompute(context.enter_step(self.body.name))

    def _execute(self, batch: CandidateBatch, context: OptimizationContext) -> CandidateBatch:
        """Execute iterations until completion or a runtime stop condition."""
        result = batch
        for index in range(self.iterations):
            if context.state.stop_requested or len(result) == 0:
                break
            previous_result = result
            iteration_result = self.body(result, context.enter_iteration(index))
            context.state.record_iteration()

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
