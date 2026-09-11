"""Lifecycle contract for every executable pipeline node.

``PipelineBuilder`` creates a tree whose composite and leaf nodes all inherit
from :class:`Step`. ``PepCompassPipeline.run`` invokes its root. The public
``__call__`` method supplies iteration preparation, tracking, resource
sampling, timing, and failure reporting before delegating only the actual
transformation to :meth:`_execute`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

from pep_compass.data.optimization import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class Step(ABC):
    """Transform a candidate batch within the common execution lifecycle.

    Subclasses normally implement only 
    * :meth:`_execute`. 
    Override
    * :meth:`precompute` for state independent of the current batch,
    * :meth:`prepare_iteration` for state that must be refreshed immediately before each call. 
    
    Subclasses **must not** override :meth:`__call__`, because
    doing so would bypass tracking, memory sampling, timing, and error events.

    Composite subclasses are 
    * ``Flow``, 
    * ``Loop``, 
    * ``Parallel`` 
    under ``optimization.engine.operations``. Leaf subclasses are component-family
    base classes under ``optimization.components``.
    """

    @property
    def name(self) -> str:
        """Return the stable tracking name of this step."""
        return self.__class__.__name__

    def precompute(self, context: OptimizationContext) -> None:
        """Prepare reusable state once before the optimization run.

        :param context: Runtime services shared by all steps.
        :type context: OptimizationContext
        """

    def prepare_iteration(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> None:
        """Refresh state that depends on the current candidate batch."""

    def __call__(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Execute the fixed lifecycle around the subclass transformation."""
        # Execution scope and batch-dependent preparation
        step_context = context.enter_step(self.name)
        self.prepare_iteration(batch, step_context)

        # Before-step diagnostics and tracking
        step_context.stability_monitor.sample(
            f"step.before:{'/'.join(step_context.scope.path)}",
            batch,
            step_context.state,
        )
        enabled = step_context.tracker.is_enabled(step_context.scope)
        handle = None
        if enabled:
            handle = step_context.tracker.begin_step(
                self, batch, step_context.scope, step_context
            )
        started_at = perf_counter()

        # Component or composite operation
        try:
            result = self._execute(batch, step_context)
        except Exception as error:
            if enabled:
                step_context.tracker.fail_step(
                    handle, self, batch, step_context.scope, step_context, error
                )
            raise
        duration_seconds = perf_counter() - started_at

        # After-step tracking, diagnostics, and structured debug log
        if enabled:
            step_context.tracker.end_step(
                handle,
                self,
                batch,
                result,
                step_context.scope,
                step_context,
            )
        step_context.stability_monitor.sample(
            f"step.after:{'/'.join(step_context.scope.path)}",
            result,
            step_context.state,
        )
        logger.debug(
            "Step path=%s input_candidates=%s output_candidates=%s "
            "duration_seconds=%.6f.",
            "/".join(step_context.scope.path),
            len(batch),
            len(result),
            duration_seconds,
        )
        return result

    @abstractmethod
    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Implement the batch transformation in a concrete step."""
