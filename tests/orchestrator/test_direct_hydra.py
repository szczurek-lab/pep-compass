"""Hydra targets, override groups and wrapper-only execution."""

import json
from dataclasses import replace

import pytest
from hydra.utils import instantiate

from pep_compass import Pipeline
from pep_compass.optimization.saving.reader import AnalysisReader
from pep_compass.orchestrator.configuration import compose_configuration
from pep_compass.orchestrator.planning import load_plan, plan_configuration, save_plan
from pep_compass.orchestrator.runner import Orchestrator


def configuration(tmp_path, backend="local"):
    return {
        "pipeline": {"_target_": "tests.fixtures.orchestration.create", "count": 2},
        "backend": {
            "_target_": "pep_compass.orchestrator.backends."
            + backend
            + "."
            + {
                "local": "LocalBackend",
                "subprocess": "SubprocessBackend",
                "slurm": "SlurmBackend",
            }[backend]
        },
        "input": {"sequences": ["AA", "CC"], "mode": "individual"},
        "orchestrator": {"name": "test", "output_dir": str(tmp_path)},
        "runner": {"storage": {"backend": "sqlite"}},
    }


def test_defaults_groups_and_constructor_overrides_do_not_load_weights():
    config = compose_configuration(
        [
            "pipeline.mutation_candidates=8",
            "backend=slurm",
            "+backend.settings.time=00:01:00",
        ]
    )
    pipeline = instantiate(config.pipeline, _convert_="all")
    assert isinstance(pipeline, Pipeline)
    assert (
        pipeline.to_schema()
        .steps[0]
        .children["mutation_generator"]
        .parameters["maximum_candidates"]
        == 8
    )
    assert config.backend._target_.endswith("SlurmBackend")
    assert pipeline.validate() == []


def test_arbitrary_public_component_targets_build_same_python_objects():
    values = {
        "_target_": "pep_compass.Pipeline",
        "autoencoder": {"_target_": "tests.fixtures.orchestration.FixtureResource"},
        "steps": [
            {"_target_": "pep_compass.steps.filters.CandidateSubset", "count": 2},
            {"_target_": "pep_compass.steps.oracles.Hydrophobicity"},
        ],
    }
    pipeline = instantiate(values, _convert_="all")
    result = pipeline.run(["AA", "CC", "DD"], seed=7)
    assert len(result.reader.get_archive()) == 2
    result.close()


def test_frozen_plan_roundtrip_and_tampering(tmp_path):
    plan = plan_configuration(configuration(tmp_path))
    saved, path = save_plan(plan)
    restored = load_plan(path)
    assert restored.to_dict() == saved.to_dict()
    payload = json.loads(path.read_text())
    payload["payload"]["entries"][0]["sequences"] = ["MUTATED"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity"):
        load_plan(path)


@pytest.mark.parametrize("backend", ["local", "subprocess"])
def test_dispatch_executes_public_wrapper_and_validates_completed_runs(
    tmp_path, backend
):
    plan = plan_configuration(configuration(tmp_path, backend))
    records = Orchestrator().run(plan)
    assert [record.status for record in records] == ["completed", "completed"]
    for record in records:
        with AnalysisReader(record.path) as reader:
            assert len(reader.get_archive()) == 1
    repeated = Orchestrator().run(
        replace(plan, options=replace(plan.options, skip_completed=True))
    )
    assert [record.status for record in repeated] == ["skipped", "skipped"]


def test_server_owned_slurm_template_is_frozen_before_dispatch(tmp_path):
    values = configuration(tmp_path, "slurm")
    template = tmp_path / "site.sh"
    template.write_text(
        "#!/bin/bash\n#SBATCH --array=0-${array_max}\n# site-specific initialization\n${command}\n"
    )
    values["backend"]["template_path"] = str(template)
    plan = plan_configuration(values)
    template.unlink()
    script = Orchestrator().run(plan)
    assert "site-specific initialization" in script.read_text()
    assert "SLURM_ARRAY_TASK_ID" in script.read_text()
    assert "worker" in script.read_text()
