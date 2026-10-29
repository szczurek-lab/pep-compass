"""Shared fixtures for the published-model evaluation tests.

The synthetic predictors defined here have analytically known behaviour, so
the scoring and sensitivity code can be tested without loading a published
artifact or a representation backbone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(MODULE_SOURCE) not in sys.path:
    sys.path.insert(0, str(MODULE_SOURCE))

from model_benchmark.predictors.base import HalfLifePredictor, PredictorMetadata  # noqa: E402

RESIDUE_WEIGHTS = {"K": 0.5, "R": 0.4, "W": -0.3, "G": 0.1}


def _metadata(name, supports_gradient=False):
    """Build metadata for a synthetic predictor.

    :param str name: Predictor identifier.
    :param bool supports_gradient: Whether the predictor exposes gradients.
    :return: Metadata record.
    :rtype: PredictorMetadata
    """
    return PredictorMetadata(
        name=name,
        publication="synthetic fixture",
        artifact="none",
        representation="test",
        architecture="analytic",
        training_records="0",
        native_units="log10 seconds",
        supports_gradient=supports_gradient,
    )


class CompositionPredictor(HalfLifePredictor):
    """Predictor reading composition only, so reordering changes nothing."""

    default_batch_size = 4

    def __init__(self):
        """Initialize the composition-only fixture.

        :rtype: None
        """
        super().__init__(_metadata("composition_only"))

    def _predict_log10_seconds(self, sequences):
        """Score peptides from their residue counts alone.

        :param list[str] sequences: Peptide sequences.
        :return: Predictions in ``log10`` seconds, shape ``(B,)``.
        :rtype: numpy.ndarray
        """
        return np.array(
            [3.0 + sum(RESIDUE_WEIGHTS.get(residue, 0.0) for residue in sequence) for sequence in sequences],
            dtype=float,
        )  # (B,)


class PositionalPredictor(HalfLifePredictor):
    """Predictor weighting residues by position, so order matters."""

    default_batch_size = 4

    def __init__(self):
        """Initialize the positional fixture.

        :rtype: None
        """
        super().__init__(_metadata("positional"))

    def _predict_log10_seconds(self, sequences):
        """Score peptides with position-dependent residue weights.

        :param list[str] sequences: Peptide sequences.
        :return: Predictions in ``log10`` seconds, shape ``(B,)``.
        :rtype: numpy.ndarray
        """
        scores = []
        for sequence in sequences:
            scores.append(
                3.0
                + sum(
                    (position + 1) * RESIDUE_WEIGHTS.get(residue, 0.0) / len(sequence)
                    for position, residue in enumerate(sequence)
                )
            )
        return np.asarray(scores, dtype=float)  # (B,)


class ConstantPredictor(HalfLifePredictor):
    """Predictor ignoring its input entirely."""

    default_batch_size = 8

    def __init__(self, value=3.5):
        """Initialize the constant fixture.

        :param float value: Prediction returned for every peptide.
        :rtype: None
        """
        self.value = value
        super().__init__(_metadata("constant"))

    def _predict_log10_seconds(self, sequences):
        """Return the same prediction for every peptide.

        :param list[str] sequences: Peptide sequences.
        :return: Constant predictions, shape ``(B,)``.
        :rtype: numpy.ndarray
        """
        return np.full(len(sequences), self.value, dtype=float)  # (B,)


@pytest.fixture
def composition_predictor():
    """Provide the composition-only predictor.

    :return: Predictor fixture.
    :rtype: CompositionPredictor
    """
    return CompositionPredictor()


@pytest.fixture
def positional_predictor():
    """Provide the position-aware predictor.

    :return: Predictor fixture.
    :rtype: PositionalPredictor
    """
    return PositionalPredictor()


@pytest.fixture
def constant_predictor():
    """Provide the constant predictor.

    :return: Predictor fixture.
    :rtype: ConstantPredictor
    """
    return ConstantPredictor()


@pytest.fixture
def synthetic_cohort():
    """Build a small cohort with exactly known targets and classes.

    :return: Cohort frame with the columns the scoring code requires.
    :rtype: pandas.DataFrame
    """
    sequences = ["GLFDIIK", "KLAKLAK", "YGGFLRR", "RGDSPAW", "WWKKGGL", "AAGGKKW"]
    ## REMARK: Half-lives are placed deliberately: two in each duration class
    ## of the contract (below 30 min, up to 6 h, above 6 h).
    seconds = [600.0, 1200.0, 3600.0, 10800.0, 36000.0, 90000.0]
    frame = pd.DataFrame(
        {
            "sequence_id": [f"seq{index}" for index in range(len(sequences))],
            "sequence": sequences,
            "length": [len(sequence) for sequence in sequences],
            "half_life_seconds": seconds,
            "log10_half_life": np.log10(seconds),
            "log10_spread": [0.0, 0.1, 0.0, 0.2, 0.0, 0.0],
            "record_count": [1, 2, 1, 2, 1, 1],
            "modification_free": [True] * len(sequences),
            "sources": ["synthetic"] * len(sequences),
            "duration_class": ["short", "short", "medium", "medium", "long", "long"],
        }
    )
    return frame
