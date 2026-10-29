"""Tests of the PeptiVerse adapter that need no model weights.

Loading an artifact requires the downloaded snapshot and a backbone, so the
tests here cover the parts that encode published facts and conventions: the
variant manifest, the output conventions, the conformal quantile recovery and
the sequence-to-SMILES conversion.

The agreement of the adapter with the authors' own inference script is
established by ``scripts/compare_with_upstream.py``, whose stored report is
checked here when it is present.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from model_benchmark.config import PEPTIVERSE_DIR, RESULTS_DIR
from model_benchmark.predictors.peptiverse import (
    VARIANTS,
    PeptiVersePredictor,
    _conformal_quantile,
    load_conformal_calibration,
    sequence_to_smiles,
)

SNAPSHOT_PRESENT = (PEPTIVERSE_DIR / "training_classifiers/half_life").is_dir()


def test_variant_manifest_matches_published_folders():
    """Every declared variant names a distinct published artifact directory."""
    assert len(VARIANTS) == 8
    folders = [variant.folder for variant in VARIANTS.values()]
    assert len(set(folders)) == len(folders)
    assert {variant.estimator for variant in VARIANTS.values()} == {"transformer", "xgb"}
    assert {variant.backbone for variant in VARIANTS.values()} == {"esm", "peptideclm", "chemberta"}


def test_only_sequence_artifacts_use_the_log_target():
    """The ``log1p`` target occurs exactly for the two sequence artifacts named log."""
    logarithmic = {key for key, variant in VARIANTS.items() if variant.target == "log1p"}
    assert logarithmic == {"peptiverse_transformer_esm_log", "peptiverse_xgb_esm_log"}


def test_conformal_quantile_uses_the_finite_sample_level():
    """The half-width is the finite-sample conformal quantile of absolute scores."""
    scores = np.arange(1.0, 101.0)  # (100,)
    quantile, level = _conformal_quantile(scores, alpha=0.1)
    assert level == pytest.approx(math.ceil(101 * 0.9) / 100)
    assert quantile == pytest.approx(np.quantile(scores, level))


def test_conformal_quantile_rejects_empty_scores():
    """Calibration without finite scores is an error, not a zero-width interval."""
    with pytest.raises(ValueError):
        _conformal_quantile(np.array([np.nan, np.inf]), alpha=0.1)


def test_missing_calibration_returns_none(tmp_path):
    """An absent calibration bundle yields no interval rather than raising."""
    assert load_conformal_calibration(tmp_path / "absent.joblib") is None


def test_sequence_to_smiles_is_canonical_and_distinguishes_substitutions():
    """Conversion is deterministic and sensitive to a single substitution."""
    first = sequence_to_smiles("GLFDIIK")
    assert first == sequence_to_smiles("GLFDIIK")
    assert first != sequence_to_smiles("GLFDIIR")


def test_sequence_to_smiles_rejects_non_peptides():
    """A non-peptide string is refused instead of silently returning None."""
    with pytest.raises(ValueError):
        sequence_to_smiles("123")


def test_unknown_variant_is_rejected():
    """An unknown artifact key is an error."""
    with pytest.raises(KeyError):
        PeptiVersePredictor("peptiverse_unknown_model")


@pytest.mark.skipif(not SNAPSHOT_PRESENT, reason="PeptiVerse snapshot not downloaded")
def test_unknown_output_convention_is_rejected():
    """Only the two documented output conventions are accepted."""
    with pytest.raises(ValueError):
        PeptiVersePredictor("peptiverse_transformer_esm_log", output_convention="hours")


@pytest.mark.skipif(not SNAPSHOT_PRESENT, reason="PeptiVerse snapshot not downloaded")
def test_published_calibration_is_recovered_for_sequence_artifacts():
    """The conformal half-width of the ESM artifacts is recovered from the bundle."""
    bundle = load_conformal_calibration(
        PEPTIVERSE_DIR / "training_classifiers/half_life/transformer_wt_log/mapie_calibration.joblib"
    )
    assert bundle is not None
    ## REMARK: These bundles are pickled MAPIE objects, which the published
    ## inference script rejects; the recovered width is positive and was fitted
    ## on the 130 training records.
    assert bundle["format"] == "mapie_object"
    assert bundle["quantile"] > 0
    assert bundle["calibration_size"] == 130
    assert bundle["accepted_by_published_loader"] is False


def test_upstream_comparison_report_shows_exact_agreement():
    """If the upstream comparison has been run, it must show exact agreement."""
    path = RESULTS_DIR / "upstream_comparison.json"
    if not path.is_file():
        pytest.skip("Upstream comparison has not been run")
    report = json.loads(path.read_text(encoding="utf-8"))
    ## REMARK: Under the deployed output convention the adapter must reproduce
    ## the published script to floating-point precision; any drift here would
    ## invalidate every evaluation result in this module.
    assert report["agrees_within_one_percent"] is True
    assert report["max_relative_difference"] < 1e-9
