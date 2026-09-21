"""Tests for the proteolytic event geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pep_compass.optimization.components.helpers.proteolysis import (
    MEROPS_AMINO_ACIDS,
    MEROPS_SUBSITES,
)
from pep_compass.optimization.components.helpers.proteolysis.geometry import (
    EventGeometry,
    allowed_cut_positions,
    enumerate_window_scores,
    geometry_from_ec,
    geometry_from_name,
    implied_cleavage_bond,
    masked_score_from_indices,
    masked_window_score,
    resolve_geometry,
    subsite_observability_mask,
    window_residue_indices,
)


class TestGeometryResolution:
    """Assignment of an event geometry to a panel entry."""

    @pytest.mark.parametrize(
        ("ec_number", "expected"),
        [
            ("3.4.21.5", EventGeometry.ENDOPEPTIDASE),
            ("3.4.24.17", EventGeometry.ENDOPEPTIDASE),
            ("3.4.11.2", EventGeometry.AMINOPEPTIDASE),
            ("3.4.14.5", EventGeometry.DIPEPTIDYL_PEPTIDASE),
            ("3.4.15.1", EventGeometry.PEPTIDYL_DIPEPTIDASE),
            ("3.4.17.20", EventGeometry.CARBOXYPEPTIDASE),
        ],
    )
    def test_ec_subsubclass_drives_the_geometry(self, ec_number: str, expected: EventGeometry) -> None:
        """The EC peptidase hierarchy is organized by attack geometry."""

        assert geometry_from_ec(ec_number) == expected

    @pytest.mark.parametrize("ec_number", ["3.4.14.9", "3.4.14.10"])
    def test_tripeptidyl_members_override_their_subsubclass(self, ec_number: str) -> None:
        """EC 3.4.14 mixes two geometries, so its tripeptidyl members are listed.

        Releasing three residues instead of two is a different state transition,
        so the distinction cannot be left to the sub-subclass.
        """

        assert geometry_from_ec(ec_number) is EventGeometry.TRIPEPTIDYL_PEPTIDASE
        assert geometry_from_ec("3.4.14.5") is EventGeometry.DIPEPTIDYL_PEPTIDASE

    @pytest.mark.parametrize("ec_number", ["", None, "3.4", "not-an-ec", "1.1.1.1"])
    def test_malformed_or_foreign_ec_is_unknown(self, ec_number: str | None) -> None:
        """A missing or non-peptidase EC number never guesses a geometry."""

        assert geometry_from_ec(ec_number) is EventGeometry.UNKNOWN

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("dipeptidyl-peptidase IV", EventGeometry.DIPEPTIDYL_PEPTIDASE),
            ("tripeptidyl-peptidase I", EventGeometry.TRIPEPTIDYL_PEPTIDASE),
            ("carboxypeptidase N catalytic chain", EventGeometry.CARBOXYPEPTIDASE),
            ("aminopeptidase N", EventGeometry.AMINOPEPTIDASE),
            ("matrix metallopeptidase-2", EventGeometry.ENDOPEPTIDASE),
            ("granzyme B ({Homo sapiens}-type)", EventGeometry.ENDOPEPTIDASE),
        ],
    )
    def test_name_fallback_matches_the_specific_rule_first(self, name: str, expected: EventGeometry) -> None:
        """Name rules resolve the more specific peptidyl families before the generic ones."""

        assert geometry_from_name(name) == expected

    def test_unmatched_name_stays_unknown(self) -> None:
        """An unrecognized name is not silently promoted to an endopeptidase."""

        assert geometry_from_name("protein of unclear activity") is EventGeometry.UNKNOWN

    def test_resolution_prefers_ec_and_reports_its_source(self) -> None:
        """EC wins over the name, and the provenance of the decision is returned."""

        geometry, source = resolve_geometry("3.4.11.2", "matrix metallopeptidase-2")
        assert geometry is EventGeometry.AMINOPEPTIDASE
        assert source == "ec"

        geometry, source = resolve_geometry("", "carboxypeptidase B2")
        assert geometry is EventGeometry.CARBOXYPEPTIDASE
        assert source == "name"

        geometry, source = resolve_geometry("", "")
        assert geometry is EventGeometry.UNKNOWN
        assert source == "unresolved"

    def test_excluded_entries_are_marked_non_hydrolytic(self) -> None:
        """A manifest exclusion overrides any EC-implied geometry."""

        geometry, source = resolve_geometry(
            "2.3.2.13", "coagulation factor XIIIa", include_in_hydrolysis_model=False
        )
        assert geometry is EventGeometry.NON_HYDROLASE
        assert source == "manifest_exclusion"


class TestAllowedCutPositions:
    """Bonds a peptidase of a given geometry may attack."""

    def test_endopeptidase_may_cut_every_bond(self) -> None:
        """Every internal bond is a candidate for an endopeptidase."""

        mask = allowed_cut_positions(10, EventGeometry.ENDOPEPTIDASE)
        assert mask.shape == (9,)
        assert mask.all()

    @pytest.mark.parametrize(
        ("geometry", "expected_bond"),
        [
            (EventGeometry.AMINOPEPTIDASE, 0),
            (EventGeometry.DIPEPTIDYL_PEPTIDASE, 1),
            (EventGeometry.TRIPEPTIDYL_PEPTIDASE, 2),
        ],
    )
    def test_n_terminal_release_uses_one_bond(self, geometry: EventGeometry, expected_bond: int) -> None:
        """An N-terminal exopeptidase has exactly one admissible bond."""

        mask = allowed_cut_positions(12, geometry)
        assert mask.sum() == 1
        assert mask[expected_bond]

    @pytest.mark.parametrize(
        ("geometry", "offset_from_end"),
        [(EventGeometry.CARBOXYPEPTIDASE, 1), (EventGeometry.PEPTIDYL_DIPEPTIDASE, 2)],
    )
    def test_c_terminal_release_uses_one_bond(self, geometry: EventGeometry, offset_from_end: int) -> None:
        """A C-terminal exopeptidase attacks the matching bond from the C-terminus."""

        length = 12
        mask = allowed_cut_positions(length, geometry)
        assert mask.sum() == 1
        assert mask[length - 1 - offset_from_end]

    def test_internal_bonds_are_forbidden_for_exopeptidases(self) -> None:
        """Internal positions are forbidden events, not weak sites."""

        mask = allowed_cut_positions(12, EventGeometry.DIPEPTIDYL_PEPTIDASE)
        assert not mask[2:].any()
        assert not mask[0]

    def test_peptide_shorter_than_the_released_fragment_has_no_event(self) -> None:
        """A dipeptidyl-peptidase cannot act on a dipeptide, which leaves no remainder."""

        assert not allowed_cut_positions(2, EventGeometry.DIPEPTIDYL_PEPTIDASE).any()
        assert allowed_cut_positions(3, EventGeometry.DIPEPTIDYL_PEPTIDASE).sum() == 1

    def test_unknown_and_non_hydrolytic_geometries_produce_no_events(self) -> None:
        """Unresolved entries are never scored by default."""

        assert not allowed_cut_positions(10, EventGeometry.UNKNOWN).any()
        assert not allowed_cut_positions(10, EventGeometry.NON_HYDROLASE).any()

    def test_dipeptidase_acts_only_on_a_dipeptide(self) -> None:
        """A dipeptidase hydrolyses a free dipeptide and nothing longer."""

        assert allowed_cut_positions(2, EventGeometry.DIPEPTIDASE).tolist() == [True]
        assert not allowed_cut_positions(6, EventGeometry.DIPEPTIDASE).any()

    def test_single_residue_has_no_bonds(self) -> None:
        """A one-residue peptide has no bond to cut."""

        assert allowed_cut_positions(1, EventGeometry.ENDOPEPTIDASE).shape == (0,)

    def test_negative_length_is_rejected(self) -> None:
        """A negative length is a programming error, not an empty peptide."""

        with pytest.raises(ValueError):
            allowed_cut_positions(-1, EventGeometry.ENDOPEPTIDASE)


class TestSubsiteObservability:
    """Which subsites of a bond exist inside the peptide."""

    def test_central_bond_observes_the_full_window(self) -> None:
        """A bond far from both termini has all eight subsites."""

        mask = subsite_observability_mask(20, 9)
        assert mask.shape == (len(MEROPS_SUBSITES),)
        assert mask.all()

    def test_terminal_bonds_lose_the_outer_subsites(self) -> None:
        """The first bond has no P4/P3/P2, the last has no P2'/P3'/P4'."""

        first = subsite_observability_mask(10, 0)
        assert first.tolist() == [False, False, False, True, True, True, True, True]

        last = subsite_observability_mask(10, 8)
        assert last.tolist() == [True, True, True, True, True, False, False, False]

    def test_window_indices_hide_absent_subsites(self) -> None:
        """Residue indices outside the peptide are reported as -1."""

        indices = window_residue_indices(10, 0)
        assert indices.tolist() == [-1, -1, -1, 0, 1, 2, 3, 4]

    def test_bond_outside_the_peptide_is_rejected(self) -> None:
        """An out-of-range bond raises rather than returning a silent mask."""

        with pytest.raises(ValueError):
            subsite_observability_mask(5, 4)


class TestMaskedWindowScore:
    """Masked window score over observable subsites only."""

    @staticmethod
    def _matrix_favouring(residue: str, subsite: int) -> np.ndarray:
        """Build a matrix that prefers one residue at one subsite.

        :param residue: Preferred residue.
        :param subsite: Subsite index that carries the preference.
        :return: Log-probability matrix, shape ``(P, A)``.
        """

        probabilities = np.full((len(MEROPS_SUBSITES), len(MEROPS_AMINO_ACIDS)), 1.0 / len(MEROPS_AMINO_ACIDS))
        column = MEROPS_AMINO_ACIDS.index(residue)
        probabilities[subsite] = 0.01 / (len(MEROPS_AMINO_ACIDS) - 1)
        probabilities[subsite, column] = 0.99
        return np.log(probabilities)

    def test_uniform_matrix_scores_zero_on_any_window(self) -> None:
        """A matrix equal to the background carries no log-odds signal."""

        uniform = np.full(
            (len(MEROPS_SUBSITES), len(MEROPS_AMINO_ACIDS)), np.log(1.0 / len(MEROPS_AMINO_ACIDS))
        )
        score, observed = masked_window_score("ACDEFGHIKL", 4, uniform)
        assert score == pytest.approx(0.0)
        assert observed == len(MEROPS_SUBSITES)

    def test_matching_residue_raises_the_score(self) -> None:
        """A residue matching the preferred subsite contributes positively."""

        matrix = self._matrix_favouring("K", 3)  # P1 prefers lysine
        matching, _ = masked_window_score("AAAKAAAAAA", 3, matrix)
        mismatching, _ = masked_window_score("AAAAAAAAAA", 3, matrix)
        assert matching > 0.0 > mismatching

    def test_truncated_window_counts_only_observed_subsites(self) -> None:
        """A terminal bond is scored over fewer subsites and reports how many."""

        matrix = self._matrix_favouring("K", 3)
        score, observed = masked_window_score("KAAAAAAAAA", 0, matrix)
        assert observed == 5
        assert score > 0.0

    def test_absent_subsite_contributes_nothing(self) -> None:
        """Masking is neutral: a missing subsite neither rewards nor penalizes.

        A preference placed on a subsite that does not exist for this bond must
        leave the score unchanged, unlike the background substitution used when
        scoring whole peptides.
        """

        matrix = self._matrix_favouring("K", 0)  # P4, absent at bond 0
        score, observed = masked_window_score("KAAAAAAAAA", 0, matrix)
        assert score == pytest.approx(0.0)
        assert observed == 5

    def test_residue_outside_the_alphabet_is_rejected(self) -> None:
        """Non-standard residues must be resolved upstream, not scored as zero."""

        uniform = np.full(
            (len(MEROPS_SUBSITES), len(MEROPS_AMINO_ACIDS)), np.log(1.0 / len(MEROPS_AMINO_ACIDS))
        )
        with pytest.raises(ValueError):
            masked_window_score("ACDXFGHIKL", 4, uniform)


class TestMaskedScoreFromIndices:
    """Scoring a window given explicit subsite-to-residue indices."""

    def test_matches_the_bond_convention_on_an_internal_bond(self) -> None:
        """Supplying the indices of a bond reproduces the bond-based score."""

        matrix = TestMaskedWindowScore._matrix_favouring("K", 3)
        sequence = "AAAKAAAAAA"
        by_bond = masked_window_score(sequence, 3, matrix)
        by_indices = masked_score_from_indices(sequence, window_residue_indices(len(sequence), 3), matrix)
        assert by_indices == by_bond

    def test_absent_subsites_contribute_nothing(self) -> None:
        """A window of only -1 entries scores zero over zero subsites."""

        matrix = TestMaskedWindowScore._matrix_favouring("K", 3)
        assert masked_score_from_indices("AAAK", np.full(8, -1), matrix) == (0.0, 0)

    def test_scores_a_window_that_is_not_an_internal_bond(self) -> None:
        """A reporter substrate has no internal bond but is still scoreable.

        The peptide of a chromogenic substrate can be a single residue, for which
        the internal-bond convention has no valid bond index at all.
        """

        matrix = TestMaskedWindowScore._matrix_favouring("K", 3)
        indices = np.array([-1, -1, -1, 0, -1, -1, -1, -1])
        score, observed = masked_score_from_indices("K", indices, matrix)
        assert observed == 1
        assert score > 0.0


class TestLatentCleavageSite:
    """Scoring every candidate bond when the cleaved bond was not reported."""

    def test_every_admissible_bond_is_scored(self) -> None:
        """An endopeptidase gets a score at each internal bond."""

        matrix = TestMaskedWindowScore._matrix_favouring("K", 3)
        scores, observed, admissible = enumerate_window_scores("AAAKAAAAAA", matrix)
        assert scores.shape == observed.shape == admissible.shape == (9,)
        assert admissible.all()
        assert np.isfinite(scores).all()

    def test_inadmissible_bonds_cannot_win(self) -> None:
        """A forbidden bond carries -inf so a maximum never selects it."""

        matrix = TestMaskedWindowScore._matrix_favouring("K", 3)
        scores, _, admissible = enumerate_window_scores(
            "AAAKAAAAAA", matrix, EventGeometry.DIPEPTIDYL_PEPTIDASE
        )
        assert admissible.sum() == 1
        assert np.isneginf(scores[~admissible]).all()
        assert np.isfinite(scores[admissible]).all()

    def test_implied_bond_is_the_one_the_matrix_prefers(self) -> None:
        """The reported bond places the preferred residue at the matching subsite."""

        matrix = TestMaskedWindowScore._matrix_favouring("K", 3)  # P1 prefers lysine
        bond, score, margin = implied_cleavage_bond("AAAAAKAAAA", matrix)
        assert bond == 5  # P1 is residue 5, the lysine
        assert score > 0.0
        assert margin > 0.0

    def test_margin_separates_a_clear_winner_from_a_tie(self) -> None:
        """Two equally good candidates produce a margin of zero."""

        matrix = TestMaskedWindowScore._matrix_favouring("K", 3)
        _, _, tied_margin = implied_cleavage_bond("AAKAAAAKAA", matrix)
        _, _, clear_margin = implied_cleavage_bond("AAAAAKAAAA", matrix)
        assert tied_margin == pytest.approx(0.0)
        assert clear_margin > tied_margin

    def test_geometry_restricts_the_candidate_set(self) -> None:
        """An exopeptidase can only be assigned its one admissible bond."""

        matrix = TestMaskedWindowScore._matrix_favouring("K", 3)
        bond, _, margin = implied_cleavage_bond(
            "AAAAAKAAAA", matrix, EventGeometry.DIPEPTIDYL_PEPTIDASE
        )
        assert bond == 1
        assert np.isinf(margin)

    def test_peptide_without_an_admissible_bond_returns_none(self) -> None:
        """No candidate means no assignment, rather than a default of zero."""

        matrix = TestMaskedWindowScore._matrix_favouring("K", 3)
        bond, score, _ = implied_cleavage_bond("AA", matrix, EventGeometry.UNKNOWN)
        assert bond is None
        assert np.isneginf(score)

    def test_minimum_subsite_requirement_excludes_truncated_windows(self) -> None:
        """Bonds whose window is mostly outside the peptide can be excluded."""

        matrix = TestMaskedWindowScore._matrix_favouring("K", 3)
        _, _, admissible = enumerate_window_scores("AAKAA", matrix)
        assert admissible.sum() == 4
        bond, _, _ = implied_cleavage_bond("AAKAA", matrix, min_observed_subsites=8)
        assert bond is None


class TestRestrictWindowIndices:
    """Scoring a window under a chosen subset of subsites."""

    def test_keeps_the_named_subsites_and_blanks_the_rest(self) -> None:
        """A restricted window keeps the subsite axis aligned with the matrix."""
        from pep_compass.optimization.components.helpers.proteolysis import (
            restrict_window_indices,
            window_residue_indices,
        )

        indices = window_residue_indices(20, 8)
        restricted = restrict_window_indices(indices, ("P4", "P3", "P2", "P1"))

        assert len(restricted) == len(indices)
        assert list(restricted[:4]) == list(indices[:4])
        assert set(restricted[4:]) == {-1}

    def test_an_already_absent_subsite_stays_absent(self) -> None:
        """Restriction never revives a subsite the peptide does not occupy."""
        import numpy as np

        from pep_compass.optimization.components.helpers.proteolysis import (
            restrict_window_indices,
            window_residue_indices,
        )

        indices = window_residue_indices(10, 1)
        restricted = restrict_window_indices(indices, ("P4", "P3", "P2", "P1"))

        assert np.all(restricted[indices < 0] == -1)

    def test_scoring_a_restricted_window_sums_fewer_subsites(self) -> None:
        """The restricted score is the sum over the kept subsites alone."""
        import numpy as np

        from pep_compass.optimization.components.helpers.proteolysis import (
            masked_score_from_indices,
            restrict_window_indices,
            window_residue_indices,
        )

        matrix = np.log(np.full((8, 20), 1.0 / 20.0))
        matrix[3, 0] = np.log(0.5)
        sequence = "AAAAAAAAAAAAAAAAAAAA"
        indices = window_residue_indices(len(sequence), 8)

        full_score, full_observed = masked_score_from_indices(sequence, indices, matrix)
        p1_score, p1_observed = masked_score_from_indices(
            sequence, restrict_window_indices(indices, ("P1",)), matrix
        )

        assert full_observed == 8
        assert p1_observed == 1
        assert p1_score == pytest.approx(float(matrix[3, 0] - np.log(1.0 / 20.0)))
        assert full_score == pytest.approx(p1_score)

    def test_unknown_subsite_is_rejected(self) -> None:
        """A misspelt subsite name is a caller error, not an empty restriction."""
        from pep_compass.optimization.components.helpers.proteolysis import (
            restrict_window_indices,
            window_residue_indices,
        )

        with pytest.raises(ValueError, match="Unknown subsites"):
            restrict_window_indices(window_residue_indices(20, 8), ("P5",))
