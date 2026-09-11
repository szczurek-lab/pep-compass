"""Tests for the public manually constructed PepCompass pipeline."""

import torch

from pep_compass.data.optimization import CandidateBatch, SharedField, TensorField
from pep_compass.optimization.engine.execution.state import OptimizationState
from pep_compass.optimization.engine.operations.flow import Flow
from pep_compass.optimization.pipeline import PepCompassPipeline
from fixtures.autoencoders import MockAutoencoder
from fixtures.components import SuffixStep


def test_pipeline_can_be_constructed_without_core_or_runtime() -> None:
    """The computation API must remain independent from YAML and planning."""
    pipeline = PepCompassPipeline(
        autoencoder=MockAutoencoder(),
        root=Flow([SuffixStep("X")]),
    )

    result = pipeline.run(["AA"], seed=7)

    assert result.candidates.sequences == ("AAX",)
    assert result.candidates.latent_origins.shape == (1, 2)


def test_pipeline_result_contains_complete_oracle_observation_archive() -> None:
    """Terminal output must not discard candidates evaluated in earlier iterations."""
    state = OptimizationState()
    state.record_observations(
        "apex",
        ("AA", "AB"),
        [2.0, 1.0],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    latest = CandidateBatch(
        ["AB"],
        torch.tensor([[3.0, 4.0]]),
        {
            "oracle.apex.score": TensorField(torch.tensor([1.0])),
            "oracle.apex.direction": SharedField("minimize"),
            "oracle.apex.name": SharedField("apex"),
        },
    )

    result = PepCompassPipeline._summarize(latest, state)

    assert result.candidates.sequences == ("AA", "AB")
    assert result.candidates.latent_origins.device.type == "cpu"
    assert result.best_candidate is not None
    assert result.best_candidate.sequence == "AB"
    assert result.best_score == 1.0
