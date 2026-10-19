"""Explicit candidate snapshot component using the current run saving service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pep_compass.steps.base import Step

if TYPE_CHECKING:
    from pep_compass.data_schemas.candidates import CandidateBatch
    from pep_compass.optimization.engine.execution.context import OptimizationContext


class Snapshot(Step):
    """Capture a named batch without changing its computational meaning.

    :param name: Snapshot label within the current run.
    """

    category = "save"
    required_resources = ("saver",)

    def __init__(self, name: str = "snapshot") -> None:
        """Initialize this object with the declared configuration.

        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :rtype: None
        """
        self._configure({"name": name})
        self.snapshot_name = name

    def _execute(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> CandidateBatch:
        """Execute.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        if context.saver is None:
            raise ValueError("A save component requires the Runner saving service.")
        context.saver.capture(batch, name=self.snapshot_name, context=context)
        return batch
