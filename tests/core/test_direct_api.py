"""Acceptance contracts for direct imports, builder ownership and immutable graphs."""

import importlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from pep_compass import Pipeline, PipelineBuilder, PipelineSchema
from pep_compass.autoencoder import HydrAMP
from pep_compass.steps.filters import CandidateSubset, Deduplicate, Lams
from pep_compass.steps.flow import Loop, Parallel
from pep_compass.steps.oracles import APEX, Hydrophobicity
from tests.fixtures.orchestration import FixtureResource


def test_public_exports_are_actual_objects_under_strategies():
    for family in ("filters", "flow", "mutations", "oracles", "saving", "walkers"):
        module = importlib.import_module("pep_compass.steps." + family)
        assert module.__all__
        for name in module.__all__:
            value = getattr(module, name)
            assert value.__module__.startswith(
                "pep_compass.steps." + family + ".strategies."
            )
            assert value.__name__ == name


def test_no_removed_source_trees_or_imports():
    import ast

    import pep_compass

    root = Path(pep_compass.__file__).parent
    for folder in ("registry", "data", "runtime", "optimization/components"):
        assert not any((root / folder).rglob("*.py"))
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            imports = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [v.name for v in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert not any(
                name.startswith(
                    (
                        "pep_compass.registry",
                        "pep_compass.data.",
                        "pep_compass.runtime",
                        "pep_compass.optimization.components",
                    )
                )
                for name in imports
            ), str(path)


def test_builder_owns_independent_snapshot_and_compilation():
    component = CandidateSubset(count=2)
    pipeline = Pipeline(autoencoder=FixtureResource())
    assert not isinstance(pipeline, PipelineBuilder)
    identity = pipeline.add_component(component)
    snapshot = pipeline.to_schema()
    component.count = 100
    first = pipeline.build()
    pipeline.replace_component(identity, CandidateSubset(count=1))
    second = pipeline.build()
    assert snapshot.find(identity).parameters["count"] == 2
    assert first.root.steps[0].count == 2
    assert second.root.steps[0].count == 1
    assert first.root is not second.root
    with pytest.raises(TypeError):
        snapshot.steps[0].parameters["count"] = 7


def test_placement_identity_is_independent_of_configuration():
    component = Deduplicate()
    builder = PipelineBuilder([component, component])
    first, second = builder.schema.steps
    assert first.node_id != second.node_id
    assert first.to_config() == second.to_config()


def test_stable_edits_and_nested_ownership():
    builder = PipelineBuilder([Loop(steps=[Deduplicate()], iterations=2)])
    parent, child = [node.node_id for node, _, _ in builder.schema.walk()]
    inserted = builder.insert(CandidateSubset(count=1), before=child)
    builder.replace(child, Deduplicate(key="sequence_and_latent"))
    assert builder.schema.find(child).parameters["key"] == "sequence_and_latent"
    builder.remove(inserted)
    assert [node.node_id for node, _, _ in builder.schema.walk()] == [parent, child]
    with pytest.raises(KeyError):
        builder.remove("unknown")
    assert len(list(builder.schema.walk())) == 2


def test_atomic_scopes_keep_child_identities():
    builder = PipelineBuilder()
    with builder.loop(iterations=2) as body:
        identity = body.add(Deduplicate())
        with pytest.raises(RuntimeError):
            builder.build()
    assert builder.schema.find(identity).label == "Deduplicate"
    old = builder.schema
    with pytest.raises(RuntimeError):
        with builder.loop(iterations=3) as body:
            body.add(Deduplicate())
            raise RuntimeError("cancel")
    assert builder.schema is old


@pytest.mark.parametrize("format", ["object", "json", "text", "mermaid", "svg"])
def test_graph_formats_and_round_trip(format):
    pipeline = Pipeline(
        [Loop(steps=[Deduplicate()], iterations=2)], autoencoder=HydrAMP()
    )
    value = pipeline.to_schema(format)
    if format == "object":
        assert isinstance(value, PipelineSchema)
    else:
        assert isinstance(value, str)
        assert pipeline.to_schema().steps[0].node_id in value
    restored = Pipeline.from_schema(pipeline.to_schema("json"))
    assert restored.to_schema().to_dict() == pipeline.to_schema().to_dict()
    assert 'data-kind="enter"' in pipeline.show()
    assert 'data-kind="return"' in pipeline.show()


def test_field_validation_precedes_model_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("model loaded")

    monkeypatch.setattr(HydrAMP, "build", forbidden)
    pipeline = Pipeline(
        [CandidateSubset(count=1, mode="highest", score_field="oracle.apex.score")]
    )
    with pytest.raises(ValueError, match="unavailable fields"):
        pipeline.build()


def test_shared_mutang_requires_local_context():
    from pep_compass.steps.mutations import Mutang

    with pytest.raises(ValueError):
        Pipeline([Mutang(maximum_candidates=4)]).validate()


def test_real_oracle_construction_is_lazy(tmp_path):
    oracle = APEX(models_directory=str(tmp_path))
    Pipeline([oracle]).validate()
    assert not hasattr(oracle, "black_box")
    assert oracle.contract().produced_fields[0] == "oracle.apex.score"


@pytest.mark.parametrize("count", [True, False, 0, -1, 1.5])
def test_direct_constructor_validates_counts(count):
    with pytest.raises((TypeError, ValueError)):
        CandidateSubset(count=count)


def test_descriptions_use_real_constructor_defaults_and_display():
    pipeline = Pipeline()
    table = pipeline.describe(display="pd")
    assert isinstance(table, pd.DataFrame)
    assert list(table) == ["component", "strategy", "description", "internal_stages"]
    raw = pipeline.describe_component(CandidateSubset, display="raw")
    params = {item["name"]: item for item in raw["parameters"]}
    assert params["count"]["required"]
    assert (
        params["mode"]["default"]
        == inspect.signature(CandidateSubset).parameters["mode"].default
    )
    detailed = pipeline.describe_component(CandidateSubset(count=3), display="pd")
    assert not detailed["value"].isna().all()
    assert (
        not detailed.fillna("")
        .astype(str)
        .apply(lambda col: col.str.contains(":param"))
        .any()
        .any()
    )


def test_nested_strategy_descriptions_are_from_actual_objects():
    result = Pipeline().describe_component(Lams(), display="raw")
    assert {"scoring", "selection"} <= result["nested_strategies"].keys()


def test_schema_does_not_retain_live_component_instances():
    pipeline = Pipeline(
        [Parallel({"one": [Deduplicate()], "two": [CandidateSubset(count=1)]})]
    )
    serialized = json.loads(pipeline.to_schema("json"))
    assert "target" in serialized["steps"][0]
    assert "registry" not in str(serialized).lower()
    assert PipelineSchema.from_dict(serialized).to_dict() == serialized


@pytest.mark.parametrize(
    "operators",
    __import__("itertools").product(("sequence", "loop", "parallel"), repeat=3),
)
def test_composition_agrees_with_analytical_sequence_model(operators):
    from tests.fixtures.components import SuffixStep

    expected = ["AA"]
    steps = []
    for index, operator in enumerate(operators):
        token = str(index)
        if operator == "sequence":
            steps.append(SuffixStep(token))
            expected = [s + token for s in expected]
        elif operator == "loop":
            steps.append(Loop(steps=[SuffixStep(token)], iterations=2))
            expected = [s + token + token for s in expected]
        else:
            steps.append(
                Parallel({"left": [SuffixStep(token)], "right": [SuffixStep("R")]})
            )
            expected = [s + token for s in expected] + [s + "R" for s in expected]
    result = Pipeline(steps, autoencoder=FixtureResource()).run(
        ["AA"], seed=7, storage={"reconstruction": "trajectories"}
    )
    assert list(result.reader.get_final_batch().sequences) == expected
    result.close()


def test_real_simple_preset_preserves_frozen_reference_exactly():
    from pep_compass.predefined import simple

    reference = json.loads(
        (Path(__file__).parents[1] / "mock_data/direct_reference.json").read_text()
    )
    result = simple().run(
        ["FLYKWWIRIGRLKL"], storage={"reconstruction": "trajectories"}
    )
    columns = ["objective", "sequence", "score", "direction", "evaluation_index"]
    expected = [{key: row[key] for key in columns} for row in reference["archive"]]
    assert result.reader.get_archive()[columns].to_dict("records") == expected
    assert list(result.reader.get_final_batch().sequences) == reference["terminal"]
    result.close()


def test_zero_iteration_loop_does_not_promise_fields_from_its_body():
    graph = Pipeline(
        [
            Loop(steps=[Hydrophobicity()], iterations=0),
            CandidateSubset(
                count=1, mode="highest", score_field="oracle.hydrophobicity.score"
            ),
        ]
    )
    with pytest.raises(ValueError, match="unavailable fields"):
        graph.validate()


def test_repeated_loop_checks_its_feedback_field_contract():
    from pep_compass.steps.walkers import Sorbes

    graph = Pipeline(
        [
            Hydrophobicity(),
            Loop(
                steps=[
                    CandidateSubset(
                        count=1,
                        mode="highest",
                        score_field="oracle.hydrophobicity.score",
                    ),
                    Sorbes(),
                ],
                iterations=2,
            ),
        ]
    )
    with pytest.raises(ValueError, match="unavailable fields"):
        graph.validate()


def test_all_internal_absolute_imports_point_to_existing_modules():
    import ast

    import pep_compass

    root = Path(pep_compass.__file__).parent
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [name.name for name in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for module in names:
                if module.startswith("pep_compass."):
                    relative = Path(*module.split(".")[1:])
                    assert (root / relative).is_dir() or (root / relative).with_suffix(
                        ".py"
                    ).is_file(), (path, module)


def test_nested_loop_validation_visits_each_contract_once(monkeypatch):
    from pep_compass.steps.base import Step

    calls = []
    original = Step.contract

    def counted(step):
        calls.append(type(step).__name__)
        return original(step)

    monkeypatch.setattr(Step, "contract", counted)
    step = Deduplicate()
    for _ in range(12):
        step = Loop(steps=[step], iterations=2)
    Pipeline([step]).validate()
    assert len(calls) == 14  # Twelve loops, one leaf and the implicit root sequence.


@pytest.mark.parametrize("child", ["invalid", None, 3, {}])
def test_invalid_owned_children_fail_at_construction(child):
    with pytest.raises(TypeError, match="Step"):
        Loop(steps=[child], iterations=1)


def test_cyclic_ownership_is_rejected_before_snapshot_recursion():
    from pep_compass.steps.flow.strategies.sequence import Flow

    flow = Flow([])
    flow.steps = (flow,)
    with pytest.raises(ValueError, match="acyclic"):
        Pipeline([flow])


def test_partial_mutation_metadata_is_accepted_by_capable_scorers():
    from pep_compass.steps.flow import LocalEnumeration
    from pep_compass.steps.mutations import Mutang
    from pep_compass.steps.walkers import Sorbes

    local = LocalEnumeration(
        walker=Sorbes(),
        mutation_generator=Mutang(maximum_candidates=4),
        iterations=1,
        include_walk_points=False,
    )
    graph = Pipeline([Parallel({"mutants": [local], "seeds": [Deduplicate()]}), Lams()])
    graph.validate()


def test_partial_scores_are_rejected_by_selectors_requiring_full_fields():
    graph = Pipeline(
        [
            Parallel({"scored": [Hydrophobicity()], "unscored": [Deduplicate()]}),
            CandidateSubset(
                count=1, mode="highest", score_field="oracle.hydrophobicity.score"
            ),
        ]
    )
    with pytest.raises(ValueError, match="unavailable fields"):
        graph.validate()
