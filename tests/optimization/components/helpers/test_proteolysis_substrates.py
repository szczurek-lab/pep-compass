"""Tests for resolving kinetic-database substrate descriptions."""

from __future__ import annotations

import pytest

from pep_compass.optimization.components.helpers.proteolysis.substrates import (
    CleavageSiteSource,
    SubstrateKind,
    infer_cleavage_bond,
    parse_substrate,
    reporter_window_indices,
    resolvable_subsites,
)


class TestSequenceRecovery:
    """Recovering residues from a free-text substrate name."""

    def test_one_letter_run_inside_a_labelled_substrate(self) -> None:
        """A run of standard residues between labels is read directly."""

        parse = parse_substrate("Abz-GIATFWMLMPEQ-EDDnp")
        assert parse.kind is SubstrateKind.PEPTIDE_ONE_LETTER
        assert parse.sequence == "GIATFWMLMPEQ"
        assert parse.has_n_block
        assert parse.is_fret

    def test_three_letter_tokens_are_joined_in_order(self) -> None:
        """Hyphenated three-letter residues resolve to a sequence."""

        parse = parse_substrate("Suc-Ala-Ala-Pro-Phe-pNA")
        assert parse.kind is SubstrateKind.PEPTIDE_THREE_LETTER
        assert parse.sequence == "AAPF"
        assert parse.reporter == "pna"

    def test_configuration_prefixes_are_ignored(self) -> None:
        """L-/D-/N- prefixes name the same residue."""

        assert parse_substrate("N-benzoyl-Gly-L-Arg").sequence == "GR"

    def test_single_residue_substrate_is_marked_as_such(self) -> None:
        """A one-residue chromogenic substrate is not a peptide."""

        parse = parse_substrate("L-Ala-4-nitroanilide")
        assert parse.kind is SubstrateKind.SINGLE_RESIDUE
        assert parse.sequence == "A"

    def test_protein_substrate_yields_no_sequence(self) -> None:
        """A named protein carries no recoverable residue window."""

        parse = parse_substrate("gelatin")
        assert parse.kind is SubstrateKind.MACROMOLECULE
        assert parse.sequence == ""

    @pytest.mark.parametrize("text", ["additional information", "more", "", None])
    def test_placeholders_are_unresolved(self, text: str | None) -> None:
        """Database placeholders never count as an identified substrate."""

        parse = parse_substrate(text)
        assert parse.kind is SubstrateKind.UNRESOLVED
        assert parse.sequence == ""

    def test_non_standard_residue_is_not_invented(self) -> None:
        """A substrate built on a non-standard residue stays unresolved."""

        parse = parse_substrate("L-2-NaI-7-amido-4-carbamoylmethylcoumarin")
        assert parse.sequence == ""
        assert parse.kind is SubstrateKind.UNRESOLVED


class TestCleavageSiteInference:
    """Deciding which bond was hydrolysed."""

    def test_reporter_group_fixes_the_scissile_bond(self) -> None:
        """The bond to a leaving reporter group follows the last residue."""

        parse = parse_substrate("Suc-Ala-Ala-Pro-Phe-pNA")
        bond, source = infer_cleavage_bond(parse)
        assert source is CleavageSiteSource.REPORTER_GROUP
        assert bond == len(parse.sequence) - 1

    @pytest.mark.parametrize(
        "text",
        [
            "Abz-GIATFWMLMPEQ-EDDnp",
            "o-aminobenzoyl-GFSPFRN-(N-2,4-dinitrophenyl)",
            "Abz-TPWSALQ-YNO2",
        ],
    )
    def test_fret_substrate_has_an_unreported_internal_site(self, text: str) -> None:
        """An internally quenched substrate does not disclose its cut site.

        The quencher is often written parenthetically rather than as the final
        token, so donor and quencher are detected anywhere in the description.
        """

        parse = parse_substrate(text)
        assert parse.is_fret
        bond, source = infer_cleavage_bond(parse)
        assert bond is None
        assert source is CleavageSiteSource.UNKNOWN_INTERNAL

    def test_reporter_substrate_is_not_marked_as_fret(self) -> None:
        """A chromogenic substrate has a leaving group, not a quencher pair."""

        assert not parse_substrate("Suc-Ala-Ala-Pro-Phe-pNA").is_fret

    def test_unlabelled_peptide_has_an_unknown_site(self) -> None:
        """Without a reporter the hydrolysed bond is not recoverable."""

        bond, source = infer_cleavage_bond(parse_substrate("N-benzoyl-Gly-L-Arg"))
        assert bond is None
        assert source is CleavageSiteSource.UNKNOWN_INTERNAL

    def test_substrate_without_a_sequence_is_not_applicable(self) -> None:
        """A protein substrate has no bond index to report."""

        bond, source = infer_cleavage_bond(parse_substrate("gelatin"))
        assert bond is None
        assert source is CleavageSiteSource.NOT_APPLICABLE


class TestResolvableSubsites:
    """How many subsites a resolved substrate can populate."""

    def test_reporter_substrate_fills_only_the_non_prime_side(self) -> None:
        """The prime subsites hold the reporter, not peptide residues."""

        assert resolvable_subsites(parse_substrate("Suc-Ala-Ala-Pro-Phe-pNA")) == 4

    def test_short_reporter_substrate_fills_fewer_subsites(self) -> None:
        """A single-residue substrate reaches only P1."""

        assert resolvable_subsites(parse_substrate("L-Ala-4-nitroanilide")) == 1

    @pytest.mark.parametrize(
        "text", ["Abz-GIATFWMLMPEQ-EDDnp", "N-benzoyl-Gly-L-Arg", "gelatin", "additional information"]
    )
    def test_unresolved_cut_site_populates_nothing(self, text: str) -> None:
        """No inferable bond means no window, whatever the sequence looks like."""

        assert resolvable_subsites(parse_substrate(text)) == 0


class TestReporterWindow:
    """Subsite assignment for a substrate cleaved at its reporter group."""

    def test_residues_fill_the_non_prime_side_from_p1_backwards(self) -> None:
        """P1 is the last residue and the prime side holds no residue."""

        indices = reporter_window_indices(parse_substrate("Suc-Ala-Ala-Pro-Phe-pNA"))
        assert indices.tolist() == [0, 1, 2, 3, -1, -1, -1, -1]

    def test_short_substrate_leaves_the_outer_subsites_empty(self) -> None:
        """A dipeptide substrate reaches only P2 and P1."""

        indices = reporter_window_indices(parse_substrate("Gly-Pro-4-nitroanilide"))
        assert indices.tolist() == [-1, -1, 0, 1, -1, -1, -1, -1]

    def test_single_residue_substrate_occupies_p1_only(self) -> None:
        """A one-residue substrate has no internal bond but does have a P1."""

        indices = reporter_window_indices(parse_substrate("L-Ala-4-nitroanilide"))
        assert indices.tolist() == [-1, -1, -1, 0, -1, -1, -1, -1]

    @pytest.mark.parametrize("text", ["Abz-GIATFWMLMPEQ-EDDnp", "gelatin", "N-benzoyl-Gly-L-Arg"])
    def test_substrate_without_a_reporter_yields_no_window(self, text: str) -> None:
        """Only an inferable reporter bond defines this window."""

        assert (reporter_window_indices(parse_substrate(text)) == -1).all()

    def test_index_count_matches_resolvable_subsites(self) -> None:
        """The two functions agree on how many subsites a substrate populates."""

        for text in ["Suc-Ala-Ala-Pro-Phe-pNA", "Gly-Pro-4-nitroanilide", "L-Ala-4-nitroanilide"]:
            parse = parse_substrate(text)
            assert int((reporter_window_indices(parse) >= 0).sum()) == resolvable_subsites(parse)
