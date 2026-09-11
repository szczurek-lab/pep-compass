"""Tests for runtime plan execution and result persistence."""

import json
from pathlib import Path

from pep_compass.runtime.configuration.schema import (
    AutoencoderConfiguration,
    ExperimentConfiguration,
    RuntimeConfiguration,
)
from pep_compass.runtime.output import ResultWriter
from pep_compass.runtime.planning.plan import materialize_execution_plan
from pep_compass.runtime.runner import RuntimeRunner
from tests.fixtures.workflows import MockWorkflow


def _configuration(output: Path) -> RuntimeConfiguration:
    """Return one deterministic runtime configuration."""
    return RuntimeConfiguration(
        experiment=ExperimentConfiguration(
            name="mock",
            input={"sequences": ["AA"]},
            output_directory=output,
            seed=3,
        ),
        autoencoder=AutoencoderConfiguration("mock", "default"),
        pipeline={"suffix": "Z"},
    )


def test_runner_persists_completed_result_and_stability_samples(tmp_path) -> None:
    """A completed run must leave independently readable terminal artifacts."""
    configuration = _configuration(tmp_path)
    plan = materialize_execution_plan(configuration, working_directory=tmp_path)
    writer = ResultWriter(tmp_path)

    executions = RuntimeRunner(configuration, MockWorkflow(), writer).run(plan)

    assert [execution.status for execution in executions] == ["completed"]
    directory = writer.run_directory(plan.entries[0])
    payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["candidate_count"] == 1
    assert (directory / "candidates.csv").exists()
    assert (directory / "fields.jsonl").exists()
    assert (directory / "resolved_config.json").exists()
    assert (directory / "tracking" / "stability.csv").exists()
    assert (directory / "tracking" / "replay_manifest.json").exists()
    assert (directory / "tracking" / "run.log").stat().st_size > 0
    assert (directory / "tracking" / "steps.csv").exists()


def test_runner_resume_skips_durable_completed_run(tmp_path) -> None:
    """Resume must avoid rebuilding a run with completed status."""
    configuration = _configuration(tmp_path)
    plan = materialize_execution_plan(configuration, working_directory=tmp_path)
    writer = ResultWriter(tmp_path)
    RuntimeRunner(configuration, MockWorkflow(), writer).run(plan)
    resumed = RuntimeConfiguration(
        experiment=configuration.experiment,
        autoencoder=configuration.autoencoder,
        pipeline=configuration.pipeline,
        tracking=configuration.tracking,
        execution=configuration.execution.__class__(resume=True),
    )

    executions = RuntimeRunner(resumed, MockWorkflow(), writer).run(plan)

    assert [execution.status for execution in executions] == ["skipped"]


def test_runner_captures_test_run_diagnostics_without_output(tmp_path) -> None:
    """Diagnostic execution must return scalar steps and memory in memory."""
    configuration = _configuration(tmp_path)
    plan = materialize_execution_plan(configuration, working_directory=tmp_path)

    execution = RuntimeRunner(
        configuration,
        MockWorkflow(),
        writer=None,
        capture_diagnostics=True,
    ).run(plan)[0]

    assert execution.candidate_count == 1
    assert execution.final_candidates == (("AAZ", None),)
    assert execution.step_records
    assert execution.memory_snapshots
    assert all(snapshot.batch_bytes is not None for snapshot in execution.memory_snapshots)
