"""Tests for low-overhead runtime stability snapshots."""

from pep_compass.optimization.stability_estimation import StabilityMonitor
from pep_compass.optimization.engine.execution.state import OptimizationState
from fixtures.candidates import candidate_batch


def test_monitor_records_batch_memory_without_requiring_cuda() -> None:
    """CPU monitoring must return a useful snapshot on systems without CUDA."""
    monitor = StabilityMonitor()

    snapshot = monitor.sample("pipeline.input", candidate_batch())

    assert snapshot is not None
    assert snapshot.label == "pipeline.input"
    assert snapshot.candidates == 2
    assert snapshot.batch_bytes > 0


def test_monitor_uses_cheap_samples_between_progress_thresholds() -> None:
    """Detailed batch inspection must run only at boundaries and 10% progress."""
    monitor = StabilityMonitor(detailed_fraction=0.1)
    monitor.configure_budgets(iterations=10)
    state = OptimizationState()

    cheap = monitor.sample("step.before:test", candidate_batch(), state)
    state.record_iteration()
    detailed = monitor.sample("step.after:test", candidate_batch(), state)

    assert cheap is not None and cheap.batch_bytes is None
    assert cheap.detailed is False
    assert detailed is not None and detailed.batch_bytes is not None
    assert detailed.detail_trigger == "iterations:1/10"
