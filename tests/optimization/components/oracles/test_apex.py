"""APEX model discovery, validation and inference tests."""

from pathlib import Path

import numpy as np
import pytest
import torch


def test_apex_rejects_missing_ensemble(tmp_path: Path) -> None:
    """An absent model directory must fail during oracle construction."""
    from pep_compass.optimization.components.oracles.strategies.apex.predictor import APEXPredictor

    with pytest.raises(FileNotFoundError, match="model directory not found"):
        APEXPredictor(models_directory=tmp_path)


def test_apex_rejects_incomplete_ensemble(tmp_path: Path) -> None:
    """A partially downloaded ensemble must not reach prediction."""
    from pep_compass.optimization.components.oracles.strategies.apex.predictor import APEXPredictor

    model_directory = tmp_path / "default"
    model_directory.mkdir()
    (model_directory / "APEX_partial").touch()

    with pytest.raises(FileNotFoundError, match="expected 8.*found 1"):
        APEXPredictor(models_directory=tmp_path)


def test_apex_default_ensemble_loads_and_predicts() -> None:
    """Bundled weights must execute through the common PepCompass Oracle contract."""
    import pep_compass.optimization.components.oracles.strategies  # noqa: F401
    from pep_compass.data.optimization import CandidateBatch, TensorField
    from pep_compass.optimization.components.oracles.base import Oracle
    from pep_compass.optimization.components.oracles.manager import OracleManager
    from pep_compass.optimization.engine.execution.context import OptimizationContext
    from tests.fixtures.autoencoders import MockAutoencoder

    oracle = OracleManager.build(
        "apex",
        model="default",
        device="cpu",
        mic_aggregate="mean",
        mic_bacteria=[1, 2, 3],
    )
    batch = CandidateBatch(["FLYKWWIRIGRLKL"], torch.zeros((1, 2)))  # (B=1, D=2)
    context = OptimizationContext(autoencoder=MockAutoencoder())

    result = oracle(batch, context)
    scores = result.fields["oracle.apex.score"]

    assert isinstance(oracle, Oracle)
    assert isinstance(scores, TensorField)
    assert scores.values.shape == (1,)
    assert np.isfinite(scores.values.numpy()).all()
    assert context.state.oracle_calls == 1


def test_apex_validates_public_parameters() -> None:
    """Invalid aggregation must fail before scoring begins."""
    from pep_compass.optimization.components.oracles.strategies.apex.oracle import APEXBlackBox

    with pytest.raises(ValueError, match="mic_aggregate"):
        APEXBlackBox(mic_aggregate="median")


def test_clean_and_original_apex_strategies_are_independently_registered() -> None:
    """Both APEX implementations must use shared weights and equivalent scoring."""
    import pep_compass.optimization.components.oracles.strategies  # noqa: F401
    from pep_compass.data.optimization import CandidateBatch, TensorField
    from pep_compass.optimization.components.oracles.manager import OracleManager
    from pep_compass.optimization.engine.execution.context import OptimizationContext
    from tests.fixtures.autoencoders import MockAutoencoder

    parameters = {
        "model": "default",
        "device": "cpu",
        "mic_aggregate": "mean",
        "mic_bacteria": [1, 2, 3],
    }
    batch = CandidateBatch(
        ["FLYKWWIRIGRLKL"],
        torch.zeros((1, 2)),
    )  # (B=1, D=2)
    clean = OracleManager.build("apex", **parameters)
    original = OracleManager.build("apex_original", **parameters)

    clean_result = clean(batch, OptimizationContext(MockAutoencoder()))
    original_result = original(batch, OptimizationContext(MockAutoencoder()))
    clean_scores = clean_result.fields["oracle.apex.score"]
    original_scores = original_result.fields["oracle.apex_original.score"]

    assert isinstance(clean_scores, TensorField)
    assert isinstance(original_scores, TensorField)
    torch.testing.assert_close(clean_scores.values, original_scores.values)
