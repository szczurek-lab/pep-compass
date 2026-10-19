"""Sequential execution through the public Python pipeline wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pep_compass.orchestrator.backends.base import ExecutionBackend

if TYPE_CHECKING:
    from pathlib import Path

    from pep_compass.orchestrator.planning import ExecutionPlan
    from pep_compass.orchestrator.runner import Orchestrator, RunExecution


class LocalBackend(ExecutionBackend):
    """Execute jobs in the current process, stopping according to failure policy."""

    def run(
        self, orchestrator: Orchestrator, plan: ExecutionPlan, plan_path: Path
    ) -> tuple[RunExecution, ...]:
        """Run.

        :param orchestrator: Orchestrator.
        :type orchestrator: Orchestrator
        :param plan: Plan.
        :type plan: ExecutionPlan
        :param plan_path: Plan path.
        :type plan_path: Path
        :return: Result of this operation.
        :rtype: tuple[RunExecution, ...]
        """
        records = []
        for entry in plan.entries:
            record = orchestrator.run_entry(plan, entry)
            records.append(record)
            if record.status == "failed" and not plan.options.continue_on_error:
                break
        return tuple(records)
