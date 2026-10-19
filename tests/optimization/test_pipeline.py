"""Tests for the public manually constructed PepCompass pipeline."""

from fixtures.autoencoders import MockAutoencoder
from fixtures.components import SuffixStep

from pep_compass.optimization.engine.pipeline import ExecutablePipeline
from pep_compass.steps.flow.strategies.sequence import Flow


def test_pipeline_can_be_constructed_without_core_or_runtime() -> None:
    """The computation API must remain independent from YAML and planning."""
    pipeline = ExecutablePipeline(
        autoencoder=MockAutoencoder(),
        schema=None,
        root=Flow([SuffixStep("X")]),
    )

    result = pipeline._run(["AA"], seed=7)

    assert result.sequences == ("AAX",)
    assert result.latent_origins.shape == (1, 2)
