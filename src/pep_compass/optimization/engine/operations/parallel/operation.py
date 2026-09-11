"""Execution of named branches from one shared input batch.

``PipelineBuilder`` converts configured branches or replicas into child
``Step`` objects. This operation executes every child against the same input,
then delegates output combination to a merger from ``parallel.merge``.
"""

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from typing import Literal

from pep_compass.data.optimization import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.operations.parallel.merge import BatchMerger, ConcatenateMerger
from pep_compass.optimization.engine.execution.step import Step
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)
ParallelExecution = Literal["sequential", "concurrent"]


class Parallel(Step):
    """Run independent branches from one input and merge their outputs."""

    def __init__(
        self,
        branches: Mapping[str, Step],
        *,
        execution: ParallelExecution = "sequential",
        merger: BatchMerger | None = None,
    ) -> None:
        if not branches:
            raise ValueError("Parallel requires at least one branch.")
        if execution not in {"sequential", "concurrent"}:
            raise ValueError("Parallel execution must be sequential or concurrent.")
        self.branches = dict(branches)
        self.execution = execution
        self.merger = merger or ConcatenateMerger()

    def precompute(self, context: OptimizationContext) -> None:
        """Precompute every branch with its independent execution scope."""
        for index, (name, branch) in enumerate(self.branches.items()):
            branch.precompute(context.enter_branch(name, index))

    def _execute(self, batch: CandidateBatch, context: OptimizationContext) -> CandidateBatch:
        """Execute branches sequentially or concurrently and merge outputs."""
        branch_items = list(self.branches.items())
        if self.execution == "sequential":
            outputs = [
                branch(batch, context.enter_branch(name, index))
                for index, (name, branch) in enumerate(branch_items)
            ]
        else:
            # REMARK: Branches share OptimizationState. Concurrent mutation of
            # counters and observations requires a dedicated synchronization
            # policy before this mode can be considered race-free.
            logger.debug("Executing %s optimization branches concurrently.", len(branch_items))
            with ThreadPoolExecutor(max_workers=len(branch_items)) as executor:
                futures = [
                    executor.submit(branch, batch, context.enter_branch(name, index))
                    for index, (name, branch) in enumerate(branch_items)
                ]
                outputs = [future.result() for future in futures]
        return self.merger(outputs)
