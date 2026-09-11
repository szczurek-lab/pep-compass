"""Tracking contracts consumed by ``Step.__call__`` during execution.

The engine emits lifecycle events through :class:`StepTracker`; it never
chooses a storage format. Runtime supplies ``CSVStepTracker`` when an output
directory exists, while manually constructed pipelines may use the null or
in-memory implementations defined here.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pep_compass.data.optimization import CandidateBatch
    from pep_compass.optimization.engine.execution.context import OptimizationContext
    from pep_compass.optimization.engine.execution.step import Step

TrackingLevel = Literal["short", "normal", "all"]


@dataclass(frozen=True)
class ExecutionScope:
    """Identify a step within nested flows, loops, and parallel branches."""

    path: tuple[str, ...] = ()
    loop_indices: tuple[int, ...] = ()
    branch_names: tuple[str, ...] = ()
    branch_indices: tuple[int, ...] = ()

    @property
    def depth(self) -> int:
        """Return the number of nested path elements."""
        return len(self.path)

    def child(self, name: str) -> "ExecutionScope":
        """Return a scope nested below a named step."""
        return ExecutionScope(
            path=(*self.path, name),
            loop_indices=self.loop_indices,
            branch_names=self.branch_names,
            branch_indices=self.branch_indices,
        )

    def loop_iteration(self, index: int) -> "ExecutionScope":
        """Return a scope containing one additional loop index."""
        return ExecutionScope(
            path=(*self.path, f"iteration[{index}]"),
            loop_indices=(*self.loop_indices, index),
            branch_names=self.branch_names,
            branch_indices=self.branch_indices,
        )

    def branch(self, name: str, index: int) -> "ExecutionScope":
        """Return a scope containing one additional branch identity."""
        return ExecutionScope(
            path=(*self.path, f"branch[{name}]"),
            loop_indices=self.loop_indices,
            branch_names=(*self.branch_names, name),
            branch_indices=(*self.branch_indices, index),
        )


class StepTracker(ABC):
    """Receive events emitted around each executable ``Step``.

    ``short`` is interpreted by the runtime CSV implementation as oracle-only
    output. ``normal`` stores summaries for all enabled steps. ``all`` also
    stores output candidates. ``max_depth`` can suppress events below a chosen
    nesting depth. The tracker observes batches but does not retain them unless
    its implementation explicitly copies their data.
    """

    level: TrackingLevel = "short"
    max_depth: int | None = None

    def is_enabled(self, scope: ExecutionScope) -> bool:
        """Return whether the scope passes the configured depth limit."""
        return self.max_depth is None or scope.depth <= self.max_depth

    def begin_step(
        self,
        step: "Step",
        batch: "CandidateBatch",
        scope: ExecutionScope,
        context: "OptimizationContext",
    ) -> Any:
        """Start one event and return an implementation-specific handle."""
        return None

    def end_step(
        self,
        handle: Any,
        step: "Step",
        input_batch: "CandidateBatch",
        output_batch: "CandidateBatch",
        scope: ExecutionScope,
        context: "OptimizationContext",
    ) -> None:
        """Record successful completion without taking batch ownership."""

    def fail_step(
        self,
        handle: Any,
        step: "Step",
        input_batch: "CandidateBatch",
        scope: ExecutionScope,
        context: "OptimizationContext",
        error: BaseException,
    ) -> None:
        """Record failure before the exception propagates."""

    def close(self) -> None:
        """Release external resources after one pipeline run."""


class NullStepTracker(StepTracker):
    """Disable tracking without conditional branches in the engine."""

    def is_enabled(self, scope: ExecutionScope) -> bool:
        """Return ``False`` for every scope."""
        return False


@dataclass(frozen=True)
class StepExecutionRecord:
    """Store scalar metadata for one completed or failed execution."""

    step_name: str
    path: tuple[str, ...]
    loop_indices: tuple[int, ...]
    branch_names: tuple[str, ...]
    branch_indices: tuple[int, ...]
    input_size: int
    output_size: int
    status: str = "completed"
    duration_seconds: float = 0.0
    oracle_calls_before: int = 0
    oracle_calls_after: int = 0
    generated_candidates_before: int = 0
    generated_candidates_after: int = 0
    error: str | None = None


class InMemoryStepTracker(StepTracker):
    """Retain scalar execution records for tests and interactive debugging."""

    def __init__(self, *, level: TrackingLevel = "normal", max_depth: int | None = None) -> None:
        self.level = level
        self.max_depth = max_depth
        self.records: list[StepExecutionRecord] = []
        self._lock = Lock()

    def begin_step(self, step, batch, scope, context):
        """Capture start time and run counters without retaining the batch."""
        return perf_counter(), context.state.oracle_calls, context.state.generated_candidates

    def end_step(self, handle, step, input_batch, output_batch, scope, context) -> None:
        """Append a successful scalar record."""
        self._append(handle, step, input_batch, scope, context, len(output_batch), None)

    def fail_step(self, handle, step, input_batch, scope, context, error) -> None:
        """Append a failed scalar record."""
        self._append(handle, step, input_batch, scope, context, 0, error)

    def _append(self, handle, step, input_batch, scope, context, output_size, error) -> None:
        """Normalize and append one event under the tracker lock."""
        record = StepExecutionRecord(
            step_name=step.name,
            path=scope.path,
            loop_indices=scope.loop_indices,
            branch_names=scope.branch_names,
            branch_indices=scope.branch_indices,
            input_size=len(input_batch),
            output_size=output_size,
            status="failed" if error is not None else "completed",
            duration_seconds=perf_counter() - handle[0],
            oracle_calls_before=handle[1],
            oracle_calls_after=context.state.oracle_calls,
            generated_candidates_before=handle[2],
            generated_candidates_after=context.state.generated_candidates,
            error=None if error is None else f"{type(error).__name__}: {error}",
        )
        with self._lock:
            self.records.append(record)
