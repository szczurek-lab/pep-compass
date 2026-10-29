"""Tests of the batch prediction interface shared by every adapter."""

from __future__ import annotations

import numpy as np
import pytest

from model_benchmark.predictors.base import decode_token_matrix


def test_batching_does_not_change_predictions(composition_predictor):
    """Chunked scoring must equal scoring every peptide in one call."""
    sequences = ["GLFDIIK", "KLAKLAK", "YGGFL", "RGDS", "WWKK", "AAGG", "KKRR"]
    composition_predictor.batch_size = 2
    chunked = composition_predictor.predict_log10_seconds(sequences)  # (N,)
    composition_predictor.batch_size = len(sequences)
    single_call = composition_predictor.predict_log10_seconds(sequences)  # (N,)
    np.testing.assert_allclose(chunked, single_call, atol=1e-12)


def test_prediction_shape_and_units(composition_predictor):
    """Predictions are one-dimensional and seconds are the exponentiated form."""
    sequences = ["GLFDIIK", "KLAKLAK"]
    logarithmic = composition_predictor.predict_log10_seconds(sequences)  # (2,)
    seconds = composition_predictor.predict_seconds(sequences)  # (2,)
    assert logarithmic.shape == (2,)
    np.testing.assert_allclose(seconds, 10.0**logarithmic, rtol=1e-12)


def test_oracle_call_returns_column(composition_predictor):
    """The oracle-style call accepts a padded token matrix and returns ``(B, 1)``."""
    matrix = np.array([list("GLFDIIK"), list("KLAKLA ")])  # (2, 7)
    result = composition_predictor(matrix)  # (2, 1)
    assert result.shape == (2, 1)
    direct = composition_predictor.predict_log10_seconds(["GLFDIIK", "KLAKLA"])  # (2,)
    np.testing.assert_allclose(result.ravel(), direct, atol=1e-12)


def test_token_matrix_decoding_strips_padding():
    """Padding tokens are removed when a token matrix is decoded."""
    matrix = np.array([list("GLFD  "), list("KLAKLA")])  # (2, 6)
    assert decode_token_matrix(matrix) == ["GLFD", "KLAKLA"]


def test_empty_input_is_rejected(composition_predictor):
    """An empty batch or a blank sequence is an error, not a silent result."""
    with pytest.raises(ValueError):
        composition_predictor.predict_log10_seconds([])
    with pytest.raises(ValueError):
        composition_predictor.predict_log10_seconds(["GLFD", "  "])


def test_gradient_absence_is_explicit(composition_predictor):
    """An adapter without gradients raises rather than returning zeros."""
    with pytest.raises(NotImplementedError):
        composition_predictor.per_residue_gradient("GLFDIIK")


def test_metadata_is_serializable(composition_predictor):
    """The provenance record carries the fields stored next to every result."""
    record = composition_predictor.describe()
    for field in ("name", "publication", "artifact", "representation", "architecture"):
        assert field in record
