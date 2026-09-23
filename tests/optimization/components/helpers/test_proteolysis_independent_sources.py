"""Tests for reducing an independent cleavage dataset to a comparable matrix."""

from __future__ import annotations

import numpy as np
import pytest

from pep_compass.optimization.components.helpers.proteolysis import (
    MEROPS_AMINO_ACIDS,
    MEROPS_SUBSITES,
    compare_specificity_matrices,
    log_probability_matrix,
    specificity_counts_from_windows,
    window_from_sequence,
)


class TestSpecificityCountsFromWindows:
    """Tallying residues per subsite over a set of cleavage windows."""

    def test_counts_every_resolved_position(self) -> None:
        """Each window contributes one count at each subsite it resolves."""
        counts = specificity_counts_from_windows(["AAAAKKKK", "AAAAKKKK"])

        assert counts.shape == (len(MEROPS_SUBSITES), len(MEROPS_AMINO_ACIDS))
        alanine = MEROPS_AMINO_ACIDS.index("A")
        lysine = MEROPS_AMINO_ACIDS.index("K")
        assert counts[0, alanine] == 2
        assert counts[4, lysine] == 2
        assert counts.sum() == 2 * len(MEROPS_SUBSITES)

    def test_gap_positions_contribute_nothing(self) -> None:
        """A source resolving six of eight subsites leaves two rows empty."""
        counts = specificity_counts_from_windows(["--CCKKKK", "--DDKKKK"])

        assert counts[0].sum() == 0
        assert counts[1].sum() == 0
        assert counts[2].sum() == 2

    def test_unknown_residue_is_ignored_rather_than_mapped(self) -> None:
        """A character outside the alphabet must not be silently bucketed."""
        counts = specificity_counts_from_windows(["BAAAKKKK"])

        assert counts[0].sum() == 0
        assert counts[1].sum() == 1

    def test_wrong_length_window_is_rejected(self) -> None:
        """A window of the wrong length is a parsing error, not a short window."""
        with pytest.raises(ValueError, match="length"):
            specificity_counts_from_windows(["AAAA"])


class TestLogProbabilityMatrix:
    """Turning counts into smoothed log-probabilities."""

    def test_rows_are_normalised(self) -> None:
        """Every subsite row is a probability distribution."""
        counts = specificity_counts_from_windows(["AAAAKKKK", "CCCCDDDD"])
        matrix = log_probability_matrix(counts)

        assert np.allclose(np.exp(matrix).sum(axis=1), 1.0)

    def test_empty_row_is_the_prior(self) -> None:
        """An unobserved subsite falls back to the uniform prior, not to zero."""
        counts = specificity_counts_from_windows(["--AAKKKK"])
        matrix = log_probability_matrix(counts)

        assert np.allclose(np.exp(matrix[0]), 1.0 / len(MEROPS_AMINO_ACIDS))

    def test_smoothing_keeps_unobserved_residues_finite(self) -> None:
        """A residue never seen at a subsite still has a finite log-probability."""
        counts = specificity_counts_from_windows(["AAAAKKKK"] * 50)
        matrix = log_probability_matrix(counts)

        assert np.all(np.isfinite(matrix))

    def test_non_positive_prior_is_rejected(self) -> None:
        """A zero prior would make an unobserved residue impossible."""
        with pytest.raises(ValueError, match="alpha"):
            log_probability_matrix(np.zeros((8, 20)), alpha=0.0)


class TestCompareSpecificityMatrices:
    """Comparing two matrices subsite by subsite."""

    def test_identical_matrices_agree_perfectly(self) -> None:
        """A matrix compared with itself has zero divergence and full agreement."""
        matrix = log_probability_matrix(specificity_counts_from_windows(["AAAAKKKK", "ACAAKRKK"]))
        records = compare_specificity_matrices(matrix, matrix)

        assert all(record["top_residue_agrees"] for record in records)
        assert all(record["jensen_shannon_bits"] == pytest.approx(0.0, abs=1e-12) for record in records)
        assert all(record["spearman"] == pytest.approx(1.0) for record in records)

    def test_disagreeing_matrices_are_separated(self) -> None:
        """Different preferences give a positive divergence and a different top residue."""
        first = log_probability_matrix(specificity_counts_from_windows(["AAAAAAAA"] * 40))
        second = log_probability_matrix(specificity_counts_from_windows(["KKKKKKKK"] * 40))
        records = compare_specificity_matrices(first, second)

        assert all(not record["top_residue_agrees"] for record in records)
        assert all(record["jensen_shannon_bits"] > 0.5 for record in records)

    def test_unobserved_subsites_are_reported_not_compared(self) -> None:
        """A subsite the source cannot resolve must not be scored against a prior."""
        matrix = log_probability_matrix(specificity_counts_from_windows(["--AAKKKK"] * 10))
        observed = [False, False, True, True, True, True, True, True]
        records = compare_specificity_matrices(matrix, matrix, observed)

        assert records[0]["observed"] is False
        assert np.isnan(records[0]["spearman"])
        assert records[2]["observed"] is True
        assert records[2]["spearman"] == pytest.approx(1.0)

    def test_shape_mismatch_is_rejected(self) -> None:
        """Comparing matrices of different shape is a caller error."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            compare_specificity_matrices(np.zeros((8, 20)), np.zeros((6, 20)))


class TestWindowFromSequence:
    """Reading a window around a bond."""

    def test_central_bond_gives_a_full_window(self) -> None:
        """A bond far from both termini resolves all eight subsites, P1 fourth."""
        sequence = "ACDEFGHIKLMN"
        window = window_from_sequence(sequence, 5)

        assert window == "DEFGHIKL"
        assert "-" not in window
        assert window[3] == sequence[5]

    def test_terminal_bond_is_padded_not_shifted(self) -> None:
        """Missing subsites are gaps, so P1 stays at the fourth position."""
        sequence = "ACDEFGHI"
        window = window_from_sequence(sequence, 1)

        assert window == "--ACDEFG"
        assert window[3] == sequence[1]
