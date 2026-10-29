"""Tests of the cohort construction against the processed dataset.

These tests read the repository's processed half-life tables, so they verify
the cohort rules on the real data the evaluation uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from model_benchmark.config import CANONICAL_ALPHABET, COHORTS, MAX_LENGTH, MIN_LENGTH, PRIMARY_COHORT
from model_benchmark.dataset import cohort_summary, load_cohort


@pytest.fixture(scope="module")
def cohort():
    """Load the primary cohort once for the module.

    :return: Aggregated primary cohort.
    :rtype: pandas.DataFrame
    """
    return load_cohort(PRIMARY_COHORT)


def test_cohort_respects_alphabet_and_length(cohort):
    """Every peptide uses the canonical alphabet and the configured length window."""
    assert not cohort.empty
    assert cohort.sequence.map(lambda text: set(text) <= set(CANONICAL_ALPHABET)).all()
    assert cohort.length.between(MIN_LENGTH, MAX_LENGTH).all()


def test_cohort_has_one_row_per_peptide(cohort):
    """Aggregation collapses repeated measurements of one peptide."""
    assert cohort.sequence_id.is_unique
    assert (cohort.record_count >= 1).all()


def test_aggregated_target_is_the_geometric_mean():
    """The aggregated point estimate is the geometric mean of its records."""
    records = load_cohort(PRIMARY_COHORT, aggregate=False)
    cohort = load_cohort(PRIMARY_COHORT)
    repeated = cohort.loc[cohort.record_count > 1]
    if repeated.empty:
        pytest.skip("The primary cohort holds no repeated measurements")

    row = repeated.iloc[0]
    matching = records.loc[records.sequence_id == row.sequence_id, "half_life_seconds"].to_numpy(dtype=float)
    expected = float(10.0 ** np.mean(np.log10(matching)))
    assert row.half_life_seconds == pytest.approx(expected, rel=1e-9)


def test_duration_class_matches_the_contract_thresholds(cohort):
    """Classes follow the contract thresholds of 30 minutes and 6 hours."""
    short = cohort.loc[cohort.duration_class == "short", "half_life_seconds"]
    medium = cohort.loc[cohort.duration_class == "medium", "half_life_seconds"]
    long_lived = cohort.loc[cohort.duration_class == "long", "half_life_seconds"]
    assert (short < 1800).all()
    assert medium.between(1800, 21600).all()
    assert (long_lived > 21600).all()


def test_logarithmic_target_is_consistent(cohort):
    """The logarithmic column matches the half-life column exactly."""
    np.testing.assert_allclose(
        cohort.log10_half_life.to_numpy(dtype=float),
        np.log10(cohort.half_life_seconds.to_numpy(dtype=float)),
        rtol=1e-12,
    )


def test_named_cohorts_are_nested_as_declared():
    """The primary cohort is a subset of the wider cohorts it is nested in."""
    primary = set(load_cohort("human_serum_in_vitro").sequence_id)
    any_context = set(load_cohort("human_serum_any").sequence_id)
    with_plasma = set(load_cohort("human_serum_plasma_in_vitro").sequence_id)
    all_species = set(load_cohort("blood_in_vitro_any_species").sequence_id)
    assert primary <= any_context
    assert primary <= with_plasma
    ## REMARK: The extended cohort only drops the species filter, so it
    ## contains every human in vitro blood peptide and strictly more.
    assert with_plasma <= all_species
    assert len(all_species) > len(with_plasma)


def test_extended_cohort_excludes_in_vivo_records():
    """The extended cohort measures proteolysis, so in vivo records are absent."""
    records = load_cohort("blood_in_vitro_any_species", aggregate=False)
    assert set(records.in_vivo_in_vitro.unique()) == {"in vitro"}
    assert set(records.matrix.unique()) <= {"serum", "plasma"}
    ## REMARK: Dropping the species filter is the only difference from the
    ## human cohort, so several species must be present.
    assert records.species.nunique() > 1


def test_unknown_cohort_is_rejected():
    """An unknown cohort name is an error."""
    with pytest.raises(KeyError):
        load_cohort("mouse_liver")


def test_summary_reports_every_class(cohort):
    """The cohort summary counts all three duration classes."""
    summary = cohort_summary(cohort)
    assert summary.peptides.iloc[0] == len(cohort)
    total = sum(summary[f"class_{name}"].iloc[0] for name in ("short", "medium", "long"))
    assert total == int(cohort.duration_class.notna().sum())


def test_cohort_definitions_are_complete():
    """Every declared cohort can be built."""
    for name in COHORTS:
        assert not load_cohort(name).empty
