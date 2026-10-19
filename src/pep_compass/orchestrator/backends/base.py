"""Contract for dispatch strategies using the public wrapper worker."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from pep_compass.orchestrator.planning import ExecutionPlan
    from pep_compass.orchestrator.runner import Orchestrator, RunExecution


class ExecutionBackend(ABC):
    """Dispatch independent jobs; computational execution belongs to Pipeline."""

    @abstractmethod
    def run(
        self, orchestrator: Orchestrator, plan: ExecutionPlan, plan_path: Path
    ) -> tuple[RunExecution, ...] | Path:
        """Return terminal job records or a generated scheduler-script path.

        :param orchestrator: Orchestrator.
        :type orchestrator: Orchestrator
        :param plan: Plan.
        :type plan: ExecutionPlan
        :param plan_path: Plan path.
        :type plan_path: Path
        :return: Result of this operation.
        :rtype: tuple[RunExecution, ...] | Path
        """

    @classmethod
    def resolve_parameters(
        cls, parameters: Mapping[str, object], working_directory: str | Path
    ) -> Mapping[str, object]:
        """Resolve backend-owned resources before freezing a plan.

        :param parameters: Validated factory parameters.
        :param working_directory: Base directory for configuration-relative paths.
        :return: Portable parameter mapping.

        :type parameters: Mapping[str, object]
        :type working_directory: str | Path
        :rtype: Mapping[str, object]
        """
        return parameters
