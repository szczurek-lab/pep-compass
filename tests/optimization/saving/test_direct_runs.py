"""Cross-component retention, seeded execution and replay acceptance tests."""

import json
import random

import numpy as np
import pandas as pd
import pytest
import torch

from pep_compass import Pipeline
from pep_compass.optimization.saving.reader import AnalysisReader
from pep_compass.steps.filters import CandidateSubset
from pep_compass.steps.flow import LocalEnumeration, Loop
from pep_compass.steps.oracles import Hydrophobicity
from pep_compass.steps.saving import Snapshot
from tests.fixtures.direct_components import (
    Failure,
    FirstFilter,
    LengthOracle,
    PairGenerator,
    RandomWalker,
)
from tests.fixtures.orchestration import FixtureResource


def local():
    return LocalEnumeration(
        walker=RandomWalker(),
        mutation_generator=PairGenerator(),
        iterations=2,
        trajectories=2,
        include_walk_points=False,
    )


def test_base_retains_evaluations_removed_by_downstream_selection():
    pipeline = Pipeline(
        [Hydrophobicity(), FirstFilter()], autoencoder=FixtureResource()
    )
    result = pipeline.run(
        ["AA", "CC", "DD"], seed=7, storage={"reconstruction": "trajectories"}
    )
    assert len(result.reader.get_archive()) == 3
    assert len(result.reader.get_final_batch()) == 1
    assert len(result.reader.get_evaluations()) == 3
    result.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_complete_batch_limit_never_discards_already_scored_peptides(backend, tmp_path):
    pipeline = Pipeline(
        [Loop(steps=[Hydrophobicity()], iterations=3)], autoencoder=FixtureResource()
    )
    storage = {
        "backend": backend,
        "path": str(tmp_path / "run") if backend == "sqlite" else None,
    }
    result = pipeline.run(
        ["AA", "CC", "DD"], execution={"oracle_calls": 1}, storage=storage
    )
    assert result.summary["oracle_calls"] == 3
    assert len(result.reader.get_archive()) == 3
    result.close()


def test_memory_sqlite_export_use_one_logical_schema(tmp_path):
    pipeline = Pipeline(
        [Hydrophobicity(), Snapshot("evaluated"), FirstFilter()],
        autoencoder=FixtureResource(),
    )
    memory = pipeline.run(
        ["AA", "CC"], seed=7, storage={"reconstruction": "trajectories"}
    )
    path = memory.save(tmp_path / "export")
    with AnalysisReader(path) as exported:
        pd.testing.assert_frame_equal(
            memory.reader.get_archive(), exported.get_archive()
        )
        assert exported.get_final_batch().sequences == ("AA",)
        assert exported.get_snapshot("evaluated")[0].sequences == ("AA", "CC")
    disk = pipeline.run(
        ["AA", "CC"],
        seed=7,
        storage={
            "backend": "sqlite",
            "path": str(tmp_path / "disk"),
            "reconstruction": "trajectories",
        },
    )
    pd.testing.assert_frame_equal(
        memory.reader.get_archive(), disk.reader.get_archive()
    )
    assert disk.reader.tables["evaluation_archive"].count_rows() == 2
    chunks = list(
        disk.reader.tables["evaluation_archive"].scan(["sequence"], chunk_size=1)
    )
    assert [len(chunk) for chunk in chunks] == [1, 1]
    memory.close()
    disk.close()


def test_multiple_replay_boundaries_restore_all_rngs_and_consumed_budgets():
    pipeline = Pipeline(
        [Loop(steps=[local(), LengthOracle()], iterations=2)],
        autoencoder=FixtureResource(),
    )
    result = pipeline.run(["AA"], seed=7, storage={"reconstruction": "boundaries"})
    checkpoints = result.reader._table_frame("checkpoints")
    assert len(checkpoints) == 2
    boundaries = []
    for checkpoint in checkpoints.itertuples():
        raw = result.reader.backend.load_payload(checkpoint.payload_id)
        assert "observations" not in raw["state"]
        assert "objective_directions" not in raw["state"]
        assert "observer" not in raw["state"]
        boundaries.append(raw["evaluation_count"])
        restored = result.reader._load_checkpoint(checkpoint.payload_id)
        records = result.reader.backend.execute(
            "SELECT objective,sequence,score FROM evaluation_records WHERE evaluation_index<? ORDER BY evaluation_index",
            (raw["evaluation_count"],),
        ).fetchall()
        expected = {}
        for objective, sequence, score in records:
            expected.setdefault(objective, {})[sequence] = score
        assert restored["state"]["observations"] == expected
        assert result.reader.reconstruct_local_enumeration(
            int(checkpoint.checkpoint_id)
        ).matches
    assert boundaries[0] == 0
    assert boundaries[1] > boundaries[0]
    result.close()


def test_retention_does_not_change_seeded_computation_or_caller_rng():
    pipeline = Pipeline([local(), LengthOracle()], autoencoder=FixtureResource())
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    first = pipeline.run(["AA"], seed=7)
    second = pipeline.run(
        ["AA"],
        seed=7,
        storage={
            "reconstruction": "trajectories",
            "component_data": "full",
        },
    )
    pd.testing.assert_frame_equal(
        first.reader.get_archive(), second.reader.get_archive()
    )
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    first.close()
    second.close()


@pytest.mark.parametrize("phase", ["precompute", "prepare_call", "execute"])
def test_failures_are_durable_and_report_original_phase(tmp_path, phase):
    pipeline = Pipeline([Failure(phase)], autoencoder=FixtureResource())
    path = tmp_path / phase
    with pytest.raises(RuntimeError, match="expected"):
        pipeline.run(["AA"], storage={"backend": "sqlite", "path": str(path)})
    with AnalysisReader(path) as reader:
        assert reader.summary["status"] == "failed"
        assert "expected" in reader.summary["error"]


def test_per_run_settings_do_not_modify_defaults_or_prior_result():
    pipeline = Pipeline(
        [CandidateSubset(count=1), Hydrophobicity()], autoencoder=FixtureResource()
    )
    pipeline.set_runner_settings(execution={"seed": 7})
    first = pipeline.run(["AA", "CC"], seed=9)
    first_archive = first.reader.get_archive().copy()
    second = pipeline.run(["AA", "CC"])
    assert pipeline.runner_settings.execution.seed == 7
    pd.testing.assert_frame_equal(first_archive, first.reader.get_archive())
    assert second.reader.settings["runner_settings"]["execution"]["seed"] == 7
    first.close()
    second.close()


def test_bundle_manifest_tampering_is_rejected(tmp_path):
    result = Pipeline([Hydrophobicity()], autoencoder=FixtureResource()).run(["AA"])
    path = result.save(tmp_path / "run")
    manifest = json.loads((path / "settings.json").read_text())
    manifest["inputs"] = ["CHANGED"]
    (path / "settings.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch"):
        AnalysisReader(path)
    result.close()
