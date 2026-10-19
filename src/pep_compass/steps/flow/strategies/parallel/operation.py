"""Execution of named branches from one shared input batch.

``PipelineBuilder`` converts configured branches or replicas into child
``Step`` objects. This operation executes every child against the same input,
then delegates output combination to a merger from ``parallel.merge``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Condition
from typing import TYPE_CHECKING, Literal

from pep_compass.data_schemas.candidates import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.steps.base import Step
from pep_compass.utils.logger import get_custom_logger
from pep_compass.utils.randomness import random_scope

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


logger = get_custom_logger(__name__)
ParallelExecution = Literal["sequential", "concurrent"]


class Parallel(Step):
    """Run independent branches from one input and merge their outputs."""

    display_parameters = ("execution",)
    category = "flow"
    kind = "parallel"

    def __init__(
        self,
        branches: Mapping[str, Sequence[Step]],
        *,
        execution: Literal["sequential", "threaded"] = "sequential",
    ) -> None:
        """Initialize this object with the declared configuration.

        :param branches: Branches.
        :type branches: Mapping[str, Sequence[Step]]
        :param execution: Execution settings or a mapping overriding their defaults.
        :type execution: Literal['sequential', 'threaded']
        :rtype: None
        """
        from pep_compass.steps.flow.strategies.sequence import Flow

        if not branches:
            raise ValueError("Parallel requires at least one branch.")
        if execution not in {"sequential", "concurrent"}:
            raise ValueError("Parallel execution must be sequential or concurrent.")
        self.branches = {
            key: value
            if isinstance(value, Flow)
            else Flow([value] if isinstance(value, Step) else value)
            for key, value in branches.items()
        }
        self.execution = execution
        self._configure(dict(execution=execution))

    def graph_children(
        self,
    ) -> dict[str, Step | tuple[Step, ...] | dict[str, tuple[Step, ...]]]:
        """Graph children.

        :return: Result of this operation.
        :rtype: dict[str, Step | tuple[Step, ...] | dict[str, tuple[Step, ...]]]
        """
        return {"branches": {name: flow.steps for name, flow in self.branches.items()}}

    def precompute(self, context: OptimizationContext) -> None:
        """Precompute every branch with its independent execution scope.

        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :rtype: None
        """
        for index, (name, branch) in enumerate(self.branches.items()):
            branch.precompute(context.enter_branch(name, index))

    def _execute(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> CandidateBatch:
        """Execute branches sequentially or concurrently and merge outputs.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        branch_items = list(self.branches.items())
        contexts = [
            context.enter_branch(name, index)
            for index, (name, _) in enumerate(branch_items)
        ]

        def execute(index: int) -> CandidateBatch:
            """Execute one branch with an isolated deterministic RNG scope.

            :param index: Index.
            :type index: int
            :return: Result of this operation.
            :rtype: CandidateBatch
            """
            with random_scope(contexts[index].seed, serialize=False):
                return branch_items[index][1](batch, contexts[index])

        if self.execution == "sequential":
            outputs = [execute(index) for index in range(len(branch_items))]
        else:
            # Ordered access to shared run resources
            ## REMARK: Threads share model adapters, tracking and OptimizationState.
            ## Order their critical sections to preserve RNG streams, IDs, budgets
            ## and trust regions. Use runtime subprocesses for independent parallelism.
            logger.info(
                "Ordering %s concurrent branches sharing mutable runtime services.",
                len(branch_items),
            )
            condition = Condition()
            next_branch = [0]

            def ordered_execute(index: int) -> CandidateBatch:
                """Ordered execute.

                :param index: Index.
                :type index: int
                :return: Result of this operation.
                :rtype: CandidateBatch
                """
                with condition:
                    condition.wait_for(lambda: next_branch[0] == index)
                    try:
                        return execute(index)
                    finally:
                        next_branch[0] += 1
                        condition.notify_all()

            with ThreadPoolExecutor(max_workers=len(branch_items)) as executor:
                futures = [
                    executor.submit(ordered_execute, index)
                    for index in range(len(branch_items))
                ]
                outputs = [future.result() for future in futures]
        return CandidateBatch.concatenate(outputs)
