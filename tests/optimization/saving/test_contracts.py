"""Acceptance invariants for one evaluation source and normalized provenance."""

import itertools
import subprocess
import sys

import pytest
import torch

import pep_compass.steps as ps  # ps - PipelineSteps
from pep_compass import Pipeline
from pep_compass.data_schemas.execution import StorageSettings
from tests.fixtures.components import SuffixStep
from tests.fixtures.direct_components import FirstFilter
from tests.fixtures.evaluation_contracts import (
    CountingOracle,
    CountingPredictor,
    InternalStages,
)
from tests.fixtures.orchestration import FixtureResource


@pytest.fixture(autouse=True)
def reset_predictor_calls():
    CountingPredictor.calls.clear()


def test_fresh_namespace_exports_are_static_and_do_not_initialize_models():
    source = """import inspect, sys
import pep_compass.steps as ps
from pep_compass import Pipeline, PipelineBuilder, PipelineSchema
assert all(getattr(ps, name).__all__ for name in ps.__all__)
assert inspect.getsourcefile(Pipeline).endswith('/core/wrapper/pipeline.py')
assert 'pep_compass.autoencoder.strategies.hydramp.adapter' not in sys.modules
assert 'pep_compass.steps.oracles.strategies.apex.predictor' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", source], check=True)


def test_cache_deduplicates_within_and_across_steps_and_resets_each_run():
    pipeline = Pipeline(
        [CountingOracle(), CountingOracle()], autoencoder=FixtureResource()
    )
    for _ in range(2):
        result = pipeline.run(
            ["AA", "CC", "AA"], storage={"reconstruction": "trajectories"}
        )
        assert result.summary["oracle_calls"] == 2
        assert result.reader.get_evaluations().sequence.tolist() == ["AA", "CC"]
        batch = result.reader.get_final_batch()
        assert batch.sequences == ("AA", "CC", "AA")
        assert torch.equal(
            batch.fields["oracle.length.score"].values, torch.tensor([2.0, 2.0, 2.0])
        )
        assert batch.fields["oracle.length.score"].values.dtype == torch.float32
        result.close()
    assert CountingPredictor.calls == [("AA", "CC"), ("AA", "CC")]


@pytest.mark.parametrize(
    "sequence_order", list(itertools.permutations(["A", "CC", "A"]))
)
@pytest.mark.parametrize("chunk_size", [None, 1, 2])
def test_cache_permutation_and_microbatch_partition_equal_direct_reference(
    sequence_order, chunk_size
):
    result = Pipeline(
        [CountingOracle(evaluation_batch_size=chunk_size)],
        autoencoder=FixtureResource(),
    ).run(sequence_order, storage={"reconstruction": "trajectories"})
    batch = result.reader.get_final_batch()
    actual = batch.fields["oracle.length.score"].values
    expected = torch.tensor(
        [float(len(sequence)) for sequence in sequence_order], dtype=torch.float32
    )
    assert torch.equal(actual, expected)
    assert [sequence for call in CountingPredictor.calls for sequence in call] == list(
        dict.fromkeys(sequence_order)
    )
    assert len(result.reader.get_evaluations()) == len(set(sequence_order))
    result.close()


def test_conflicting_named_objectives_are_rejected_before_scoring():
    pipeline = Pipeline(
        [CountingOracle(offset=0.0), CountingOracle(offset=1.0)],
        autoencoder=FixtureResource(),
    )
    with pytest.raises(ValueError, match="Conflicting configurations"):
        pipeline.run(["AA"])
    assert CountingPredictor.calls == []
    with pytest.raises(ValueError, match="Conflicting configurations"):
        pipeline.validate()


def test_base_archive_keeps_all_scored_sequences_and_no_unscored_ancestor_strings():
    pipeline = Pipeline(
        [SuffixStep("K"), CountingOracle(), ps.filters.CandidateSubset(count=1)],
        autoencoder=FixtureResource(),
    )
    result = pipeline.run(["AA", "CCC"])
    assert set(result.archive.sequence) == {"AAK", "CCCK"}
    peptides = {
        row[0] for row in result.reader.backend.execute("SELECT sequence FROM peptides")
    }
    assert peptides == {"AAK", "CCCK"}
    assert (
        result.reader.backend.execute(
            "SELECT COUNT(*) FROM candidate_instances WHERE peptide_id IS NULL"
        ).fetchone()[0]
        == 2
    )
    assert len(result.reader.get_lineage()) > 0
    assert result.reader.tables["tensor_payloads"].count_rows() == 0
    assert result.reader.backend.execute("PRAGMA foreign_key_check").fetchall() == []
    result.close()


def test_normalized_batch_payloads_do_not_repeat_members_or_sequence_strings():
    result = Pipeline(
        [CountingOracle(), ps.saving.Snapshot("scored"), FirstFilter()],
        autoencoder=FixtureResource(),
    ).run(["AA", "CC"], storage={"reconstruction": "trajectories"})
    backend = result.reader.backend
    assert result.reader.get_final_batch().sequences == ("AA",)
    assert result.reader.get_snapshot("scored")[0].sequences == ("AA", "CC")
    assert backend.execute("SELECT COUNT(*) FROM peptides").fetchone()[0] == 2
    assert "payload_id" not in result.reader.tables["candidate_sets"].columns
    assert "evaluated_peptides" not in result.reader.tables
    assert "observations" not in result.reader.tables
    for row in backend.execute("SELECT DISTINCT field_name FROM batch_payloads"):
        assert row[0] not in {"sequences", "lineage.candidate_id"}
    assert backend.execute("PRAGMA foreign_key_check").fetchall() == []
    result.close()


@pytest.mark.parametrize("level", ["none", "basic", "basic_and_components"])
def test_measurement_levels_preserve_identity_and_aggregate_internal_stages(level):
    result = Pipeline(
        [ps.flow.Loop(steps=[InternalStages(), CountingOracle()], iterations=2)],
        autoencoder=FixtureResource(),
    ).run(
        ["AA", "CC"], storage={"metrics_level": level, "reconstruction": "trajectories"}
    )
    backend = result.reader.backend
    executions = result.reader._table_frame("step_executions")
    measured = executions.duration_seconds.notna()
    assert not measured.any() if level == "none" else measured.all()
    intervals = result.reader._table_frame("iterations")
    assert len(intervals) == 2
    assert intervals.parent_iteration_id.isna().all()
    if level != "none":
        assert (intervals.duration_seconds >= 0).all()
        assert (intervals.step_duration_sum_seconds <= intervals.duration_seconds).all()
        assert (
            executions.memory_peak_bytes.dropna()
            >= executions.memory_start_bytes.dropna()
        ).all()
    stages = result.reader._table_frame("execution_metrics")
    if level == "basic_and_components":
        stages = stages[stages.stage_name == "row_work"]
        assert len(stages) == 2
        assert stages.input_count.tolist() == [2, 2]
    else:
        assert stages.empty
    assert result.reader.get_final_batch().sequences == ("AA", "CC")
    assert len(result.reader.get_evaluations()) == 2
    assert backend.execute("PRAGMA foreign_key_check").fetchall() == []
    result.close()


def test_removed_storage_options_fail_instead_of_silently_changing_semantics():
    for key in ["save_terminal", "save_observation_history", "profile"]:
        with pytest.raises(TypeError):
            StorageSettings(**{key: True})
    assert StorageSettings().save_lineage is True
    with pytest.raises(ValueError, match="Snapshots require"):
        Pipeline([ps.saving.Snapshot()], autoencoder=FixtureResource()).run(["AA"])


@pytest.mark.parametrize("reconstruction", ["none", "trajectories"])
def test_mixed_old_and_new_sequences_reuse_only_matching_objective_values(
    reconstruction,
):
    graph = Pipeline(
        [
            CountingOracle(),
            ps.flow.Parallel(
                {"unchanged": [ps.filters.Deduplicate()], "extended": [SuffixStep("A")]}
            ),
            CountingOracle(),
        ],
        autoencoder=FixtureResource(),
    )
    result = graph.run(["A", "AA"], storage={"reconstruction": reconstruction})
    assert CountingPredictor.calls == [("A", "AA"), ("AAA",)]
    assert result.summary["oracle_calls"] == 3
    if reconstruction == "trajectories":
        assert result.reader.get_final_batch().sequences == ("A", "AA", "AA", "AAA")
        assert result.reader.get_final_batch().fields[
            "oracle.length.score"
        ].values.tolist() == [1.0, 2.0, 2.0, 3.0]
    backend = result.reader.backend
    assert (
        backend.execute(
            "SELECT count(*) FROM candidate_instances c JOIN peptides p USING(peptide_id) WHERE p.sequence='AA'"
        ).fetchone()[0]
        == 2
    )
    assert (
        backend.execute(
            "SELECT count(*) FROM candidate_instances WHERE peptide_id IS NULL"
        ).fetchone()[0]
        == 0
    )
    assert len(result.reader.get_evaluations()) == 3
    assert set(result.archive.sequence) == {"A", "AA", "AAA"}
    result.close()


def test_duplicate_member_positions_and_broadcast_ranges_are_unambiguous():
    from tests.fixtures.evaluation_contracts import ReverseDuplicates

    graph = Pipeline(
        [ReverseDuplicates(), CountingOracle()], autoencoder=FixtureResource()
    )
    result = graph.run(["AA", "AA"], storage={"reconstruction": "trajectories"})
    backend = result.reader.backend
    placement = graph.to_schema().steps[0].node_id
    execution = backend.execute(
        "SELECT execution_id,input_batch_id,output_batch_id FROM step_executions WHERE node_id=? AND phase='execute'",
        (placement,),
    ).fetchone()
    edges = backend.execute(
        "SELECT child_start,child_end,parent_start,parent_end,mapping_kind FROM batch_lineage WHERE child_batch_id=? ORDER BY child_start",
        (execution[2],),
    ).fetchall()
    reconstructed = {}
    for start, end, parent_start, parent_end, mode in edges:
        for position in range(start, end):
            reconstructed[position] = (
                parent_start if mode == "broadcast" else parent_start + position - start
            )
    assert reconstructed == {0: 1, 1: 0, 2: 1}
    assert len(result.archive) == 1
    assert result.reader.get_final_batch().sequences == ("AA", "AA", "AA")
    result.close()


def test_nested_iteration_membership_and_failed_iteration_state(tmp_path):
    from pep_compass.optimization.saving.reader import AnalysisReader
    from tests.fixtures.direct_components import Failure

    graph = Pipeline(
        [
            ps.flow.Loop(
                steps=[ps.flow.Loop(steps=[CountingOracle()], iterations=2)],
                iterations=2,
            )
        ],
        autoencoder=FixtureResource(),
    )
    result = graph.run(["AA"])
    iterations = result.reader._table_frame("iterations")
    assert len(iterations) == 6
    assert iterations.parent_iteration_id.notna().sum() == 4
    assert (iterations.step_duration_sum_seconds <= iterations.duration_seconds).all()
    assert set(iterations.status) == {"completed"}
    result.close()
    failing = Pipeline(
        [ps.flow.Loop(steps=[Failure("execute")], iterations=2)],
        autoencoder=FixtureResource(),
    )
    with pytest.raises(RuntimeError, match="expected"):
        failing.run(
            ["AA"], storage={"backend": "sqlite", "path": str(tmp_path / "failed")}
        )
    with AnalysisReader(tmp_path / "failed") as reader:
        assert set(reader._table_frame("iterations").status) == {"failed"}
        assert not (reader._table_frame("step_executions").status == "running").any()
        assert reader.backend.execute("PRAGMA foreign_key_check").fetchall() == []


def test_internal_precompute_stages_have_real_execution_foreign_keys():
    from tests.fixtures.evaluation_contracts import PreparationStages

    result = Pipeline(
        [PreparationStages(), CountingOracle()], autoencoder=FixtureResource()
    ).run(["AA"], storage={"metrics_level": "basic_and_components"})
    rows = result.reader.backend.execute(
        "SELECT e.phase,m.stage_name FROM execution_metrics m JOIN step_executions e USING(execution_id)"
    ).fetchall()
    assert ("precompute", "prepare_resource") in [tuple(row) for row in rows]
    assert result.reader.backend.execute("PRAGMA foreign_key_check").fetchall() == []
    result.close()


def test_exhaustive_small_row_maps_preserve_duplicate_occurrences_and_range_bounds():
    from pep_compass.data_schemas.candidates import CandidateBatch, TensorField
    from pep_compass.optimization.saving.backends import SQLiteStorageBackend
    from pep_compass.optimization.saving.batches import BatchStore

    for identities in ([0, 1, 2], [0, 0, 1]):
        backend = SQLiteStorageBackend({"run_id": "mapping-test"})
        store = BatchStore(backend, True)
        source = CandidateBatch(
            ["AA", "AA", "CC"],
            torch.zeros(3, 2),
            {"lineage.candidate_id": TensorField(torch.tensor(identities))},
        )
        source_id = store.register(source, execution_id=None)
        execution_id = 0
        for length in range(5):
            for indices in itertools.product(range(3), repeat=length):
                backend.execute(
                    "INSERT INTO step_executions(execution_id,node_id,scope_json,status,phase) VALUES (?,'select','{}','completed','execute')",
                    (execution_id,),
                )
                result = (
                    source.select(indices)
                    .with_field("marker", TensorField(torch.zeros(length)))
                    .without_fields(["marker"])
                )
                batch_id = store.register(
                    result, execution_id=execution_id, parent_batch_id=source_id
                )
                edges = backend.execute(
                    "SELECT child_start,child_end,parent_batch_id,parent_start,parent_end,mapping_kind FROM batch_lineage WHERE child_batch_id=? ORDER BY child_start",
                    (batch_id,),
                ).fetchall()
                expanded = {}
                for start, end, parent, parent_start, parent_end, mode in edges:
                    assert parent == source_id
                    assert 0 <= start < end <= length
                    assert 0 <= parent_start < parent_end <= 3
                    assert parent_end - parent_start == (
                        1 if mode == "broadcast" else end - start
                    )
                    for position in range(start, end):
                        assert position not in expanded
                        expanded[position] = (
                            parent_start
                            if mode == "broadcast"
                            else parent_start + position - start
                        )
                assert expanded == dict(enumerate(indices))
                execution_id += 1
        assert backend.execute("PRAGMA foreign_key_check").fetchall() == []
        assert backend.execute("SELECT COUNT(*) FROM peptides").fetchone()[0] == 0
        backend.close()
