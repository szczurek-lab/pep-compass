"""Tests for composable engine graph operations."""

from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine import Flow, Loop, Parallel
from pep_compass.optimization.tracking import InMemoryStepTracker
from fixtures.autoencoders import MockAutoencoder
from fixtures.candidates import candidate_batch
from fixtures.components import SuffixStep


def test_parallel_defaults_to_sequential_execution() -> None:
    """Omitted execution mode must not create worker threads implicitly."""
    parallel = Parallel({"only": SuffixStep("1")})

    assert parallel.execution == "sequential"


def test_parallel_rejects_removed_auto_execution_mode() -> None:
    """The ambiguous historical auto mode must no longer be accepted."""
    import pytest

    with pytest.raises(ValueError, match="sequential or concurrent"):
        Parallel({"only": SuffixStep("1")}, execution="auto")  # type: ignore[arg-type]


def test_parallel_merges_outputs_in_declared_branch_order() -> None:
    """Concurrent completion order must not affect merged candidate order."""
    parallel = Parallel(
        {"first": SuffixStep("1"), "second": SuffixStep("2")},
        execution="concurrent",
    )

    result = parallel(
        candidate_batch(),
        OptimizationContext(MockAutoencoder(), seed=7),
    )

    assert result.sequences == ("AA1", "BB1", "AA2", "BB2")


def test_loop_stops_when_a_step_produces_an_empty_batch() -> None:
    """Empty candidate sets must not execute redundant loop iterations."""
    class EmptyStep(SuffixStep):
        calls = 0

        def _execute(self, batch, context):
            self.calls += 1
            return batch.select([])

    step = EmptyStep("")

    result = Loop(step, iterations=3)(
        candidate_batch(),
        OptimizationContext(MockAutoencoder()),
    )

    assert len(result) == 0
    assert step.calls == 1


def test_loop_preserves_last_result_when_empty_batch_requests_stop() -> None:
    """An exhaustion sentinel must not erase the last committed iteration."""
    class ExhaustionStep(SuffixStep):
        calls = 0

        def _execute(self, batch, context):
            self.calls += 1
            if self.calls == 1:
                return super()._execute(batch, context)
            context.state.stop_requested = True
            return batch.select([])

    step = ExhaustionStep("1")
    context = OptimizationContext(MockAutoencoder())

    result = Loop(step, iterations=3)(candidate_batch(), context)

    assert result.sequences == ("AA1", "BB1")
    assert step.calls == 2
    assert context.state.stop_requested is True


def test_nested_tracking_preserves_loop_and_branch_identity() -> None:
    """Tracker records must identify replicated work independently."""
    tracker = InMemoryStepTracker()
    graph = Loop(
        Parallel({"left": Flow([SuffixStep("L")])}),
        iterations=2,
    )

    graph(
        candidate_batch(),
        OptimizationContext(MockAutoencoder(), tracker=tracker),
    )

    records = [record for record in tracker.records if record.step_name == "SuffixStep"]
    assert [record.loop_indices for record in records] == [(0,), (1,)]
    assert [record.branch_names for record in records] == [("left",), ("left",)]
    assert [record.branch_indices for record in records] == [(0,), (0,)]
