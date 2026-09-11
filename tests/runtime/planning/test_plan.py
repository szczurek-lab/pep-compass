"""Tests for deterministic task and variant planning."""

from pathlib import Path

from pep_compass.runtime.configuration.schema import (
    AutoencoderConfiguration,
    ExperimentConfiguration,
    RuntimeConfiguration,
)
from pep_compass.runtime.planning.plan import materialize_execution_plan


def test_plan_combines_variants_tasks_repetitions_and_seeds() -> None:
    """Global run identity must remain stable across parameter variants."""
    configuration = RuntimeConfiguration(
        experiment=ExperimentConfiguration(
            name="test",
            input={"sequences": ["AA"], "repetitions": 2},
            seed=10,
            grid={"steps.0.loop.iterations": (1, 2)},
        ),
        autoencoder=AutoencoderConfiguration("mock", "default"),
        pipeline={"steps": [{"loop": {"iterations": 1, "steps": []}}]},
    )

    plan = materialize_execution_plan(
        configuration,
        working_directory=Path.cwd(),
    )

    assert [entry.index for entry in plan.entries] == [0, 1, 2, 3]
    assert [entry.seed for entry in plan.entries] == [10, 11, 12, 13]
    assert plan.entries[2].variant.pipeline["steps"][0]["loop"]["iterations"] == 2
