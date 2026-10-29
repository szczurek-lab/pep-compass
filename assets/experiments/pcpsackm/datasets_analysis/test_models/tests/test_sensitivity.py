"""Tests of the single-substitution analysis.

The tests establish that the analysis itself discriminates: a composition-only
predictor and a position-aware predictor must receive different verdicts, and a
predictor ignoring its input must be detected as insensitive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_benchmark.sensitivity import (
    empirical_substitution_pairs,
    mutational_scan,
    permutation_sensitivity,
    scan_summary,
    score_empirical_pairs,
)

PARENT = "GLFDIIK"


def test_scan_covers_every_substitution(composition_predictor):
    """A scan enumerates all 19 replacements at every position exactly once."""
    scan = mutational_scan(composition_predictor, PARENT)
    assert len(scan) == len(PARENT) * 19
    assert scan.position.nunique() == len(PARENT)
    assert (scan.variant_residue != scan.wild_type_residue).all()
    assert scan.variant.nunique() == len(scan)


def test_scan_rejects_non_canonical_sequences(composition_predictor):
    """A sequence outside the canonical alphabet is refused."""
    with pytest.raises(ValueError):
        mutational_scan(composition_predictor, "GLFXIIK")


def test_constant_predictor_is_detected_as_insensitive(constant_predictor):
    """A predictor ignoring its input produces identical predictions everywhere."""
    scan = mutational_scan(constant_predictor, PARENT)
    summary = scan_summary(scan, noise_floor=0.05)
    assert summary["max_absolute_delta_log10"] == pytest.approx(0.0)
    assert summary["identical_prediction_share"] == pytest.approx(1.0)
    assert summary["below_noise_floor_share"] == pytest.approx(1.0)


def test_composition_predictor_reacts_to_substitution(composition_predictor):
    """Replacing a residue changes a composition-only prediction."""
    scan = mutational_scan(composition_predictor, PARENT)
    summary = scan_summary(scan)
    assert summary["max_absolute_delta_log10"] > 0.1
    assert summary["identical_prediction_share"] < 1.0


def test_permutation_separates_composition_from_position(
    composition_predictor, positional_predictor
):
    """Reordering moves a positional predictor only."""
    composition = permutation_sensitivity(composition_predictor, PARENT, repeats=8)
    positional = permutation_sensitivity(positional_predictor, PARENT, repeats=8)
    assert composition["max_absolute_permutation_delta_log10"] == pytest.approx(0.0, abs=1e-12)
    assert positional["max_absolute_permutation_delta_log10"] > 0.0


def test_positional_variance_share_is_zero_without_position_dependence(composition_predictor):
    """A composition-only response does not depend on which position changed."""
    summary = scan_summary(mutational_scan(composition_predictor, "KKKWWWG"))
    ## REMARK: For a composition-only model the mean response is identical at
    ## every position carrying the same wild-type residue, so the between
    ## position variance comes only from differing wild-type residues.
    positional = scan_summary(mutational_scan(composition_predictor, "KKKKKKK"))
    assert positional["position_variance_share"] == pytest.approx(0.0, abs=1e-9)
    assert summary["position_variance_share"] >= 0.0


def test_measured_pairs_are_found_within_one_condition():
    """Pairs are built only from sequences measured under the same condition."""
    records = pd.DataFrame(
        {
            "sequence": ["GLFDIIK", "GLFDIIR", "GLFDIIK", "GLFDIIW"],
            "species": ["human", "human", "human", "rat"],
            "matrix": ["serum", "serum", "serum", "serum"],
            "in_vivo_in_vitro": ["in vitro", "in vitro", "in vitro", "in vitro"],
            "log10_half_life": [3.0, 3.5, 3.0, 4.0],
        }
    )
    pairs = empirical_substitution_pairs(records)
    assert len(pairs) == 1
    pair = pairs.iloc[0]
    assert {pair.parent, pair.variant} == {"GLFDIIK", "GLFDIIR"}
    assert pair.measured_delta_log10 == pytest.approx(0.5)


def test_pair_scoring_reports_direction_agreement(composition_predictor):
    """Sign agreement is computed against the measured direction."""
    pairs = pd.DataFrame(
        {
            "condition": ["human|serum|in vitro"] * 2,
            "parent": ["GLFDIIG", "GLFDIIG"],
            "variant": ["GLFDIIK", "GLFDIIW"],
            "position": [6, 6],
            "parent_residue": ["G", "G"],
            "variant_residue": ["K", "W"],
            "parent_measured_log10": [3.0, 3.0],
            "variant_measured_log10": [3.4, 2.6],
            "measured_delta_log10": [0.4, -0.4],
        }
    )
    scored, record = score_empirical_pairs(composition_predictor, pairs)
    ## REMARK: The fixture weights K positively and W negatively, matching both
    ## measured directions, so agreement must be complete.
    assert record["pairs"] == 2
    assert record["sign_agreement"] == pytest.approx(1.0)
    assert np.sign(scored.predicted_delta_log10.to_numpy()).tolist() == [1.0, -1.0]


def test_pair_scoring_filters_small_measured_effects(composition_predictor):
    """Pairs below the minimum measured effect are excluded from scoring."""
    pairs = pd.DataFrame(
        {
            "condition": ["human|serum|in vitro"],
            "parent": ["GLFDIIG"],
            "variant": ["GLFDIIK"],
            "position": [6],
            "parent_residue": ["G"],
            "variant_residue": ["K"],
            "parent_measured_log10": [3.0],
            "variant_measured_log10": [3.01],
            "measured_delta_log10": [0.01],
        }
    )
    _, record = score_empirical_pairs(composition_predictor, pairs, minimum_effect=0.1)
    assert record["pairs"] == 0
