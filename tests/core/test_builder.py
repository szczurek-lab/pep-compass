"""Tests for declarative construction of PepCompass pipelines."""

from pep_compass.core.builder import PipelineBuilder
from pep_compass.core.specification import FlowSpecification, PipelineSpecification
from fixtures.autoencoders import MockAutoencoder


def test_builder_returns_executable_pep_compass_pipeline() -> None:
    """Core must construct, but not execute, a declared pipeline."""
    pipeline = PipelineBuilder(MockAutoencoder()).build(
        PipelineSpecification(FlowSpecification(()))
    )

    result = pipeline.run(["AA"])

    assert result.candidates.sequences == ("AA",)
