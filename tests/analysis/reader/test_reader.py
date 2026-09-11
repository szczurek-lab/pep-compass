"""Tests for versioned runtime-result discovery."""

from pathlib import Path

from pep_compass.analysis.reader import ExperimentReader
from pep_compass.runtime.configuration.schema import (
    AutoencoderConfiguration,
    ExperimentConfiguration,
    RuntimeConfiguration,
)
from pep_compass.runtime.output import ResultWriter
from pep_compass.runtime.planning.plan import materialize_execution_plan
from pep_compass.runtime.runner import RuntimeRunner
from tests.fixtures.workflows import MockWorkflow


def test_reader_exposes_runtime_results_as_logical_dataset(tmp_path) -> None:
    """Reader discovery must produce both selections and ExperimentDataset."""
    configuration = RuntimeConfiguration(
        experiment=ExperimentConfiguration(
            name="reader",
            input={"sequences": ["AA"]},
        ),
        autoencoder=AutoencoderConfiguration("mock", "default"),
        pipeline={},
    )
    plan = materialize_execution_plan(configuration, working_directory=Path.cwd())
    RuntimeRunner(configuration, MockWorkflow(), ResultWriter(tmp_path)).run(plan)

    reader = ExperimentReader(tmp_path)

    assert len(reader.runs) == 1
    assert reader.dataset.schema_version == "2"
    assert reader.dataset.runs[0].identity.run_id == "run_00000"
    assert set(reader.dataset.tables) == {
        "candidates",
        "local_enumerations",
        "stability",
        "steps",
        "trajectory_points",
    }
    assert reader.dataset.tables["steps"].count_rows() == 2
    steps = reader.select().collect("steps")
    assert steps["step_name"].tolist() == ["SuffixStep", "Flow"]


def test_reader_validates_local_enumeration_replay_checkpoint(tmp_path) -> None:
    """Reader must expose SORBES points and verify reconstructed local output."""
    import torch

    from pep_compass.data.optimization import CandidateBatch
    from pep_compass.optimization.engine.operations import Flow, LocalEnumeration
    from pep_compass.optimization.pipeline import PepCompassPipeline
    from tests.fixtures.autoencoders import MockAutoencoder
    from tests.fixtures.components import SuffixStep

    class SorbesWalker(SuffixStep):
        pass

    class PairGenerator(SuffixStep):
        def _execute(self, batch, context):
            parents = torch.arange(len(batch)).repeat_interleave(2)
            expanded = batch.repeat_from_parents(parents)
            return expanded.with_sequences(
                [
                    f"{sequence}{suffix}"
                    for sequence in batch.sequences
                    for suffix in ("A", "B")
                ]
            )

    class ReplayWorkflow:
        def build_pipeline(self, pipeline_configuration, *, tracker, stability_monitor):
            operation = LocalEnumeration(
                walker=SorbesWalker("W"),
                mutation_generator=PairGenerator(""),
                filters=Flow([]),
                trajectories=2,
                trajectory_execution="batched",
                iterations=1,
                walk_time=None,
                include_walk_points=False,
            )
            return PepCompassPipeline(
                autoencoder=MockAutoencoder(),
                root=operation,
                tracker=tracker,
                stability_monitor=stability_monitor,
            )

    configuration = RuntimeConfiguration(
        experiment=ExperimentConfiguration(
            name="replay",
            input={"sequences": ["AA"]},
            seed=11,
        ),
        autoencoder=AutoencoderConfiguration("mock", "default"),
        pipeline={},
    )
    plan = materialize_execution_plan(configuration, working_directory=Path.cwd())
    RuntimeRunner(configuration, ReplayWorkflow(), ResultWriter(tmp_path)).run(plan)

    reader = ExperimentReader(tmp_path)
    replay = reader.replay("run_00000")
    checkpoint = replay.local_enumerations.iloc[0]
    execution_id = int(checkpoint["execution_id"])
    sequences, latents = replay.local_enumeration_input(execution_id)
    verification = replay.verify_local_enumeration(
        execution_id,
        replay.final_candidates["sequence"].tolist(),
    )

    assert sequences == ("AA",)
    assert latents.shape == (1, 2)
    assert len(replay.trajectory_points) == 2
    assert replay.trajectory_latents.shape == (2, 2)
    assert verification.matches
