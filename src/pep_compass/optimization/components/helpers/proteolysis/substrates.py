r"""Resolve kinetic-database substrate descriptions into scoreable peptide windows.

Kinetic databases name a substrate as free text: ``L-Ala-4-nitroanilide``,
``Abz-GIATFWMLMPEQ-EDDnp``, ``Suc-Ala-Ala-Pro-Phe-pNA`` or simply ``gelatin``.
Before a MEROPS window score can be attached to such a measurement, two separate
questions must be answered, and conflating them is the main way a coverage audit
overstates itself:

1. **Is the sequence resolvable?** Can the residues be recovered unambiguously
   from the description (:func:`parse_substrate`)?
2. **Is the cleavage site resolvable?** Even with the residues in hand, the bond
   that was actually hydrolysed is usually not reported
   (:func:`infer_cleavage_bond`).

A chromogenic or fluorogenic substrate answers the second question by
construction: the reporter group leaves from the C-terminus, so the scissile bond
is the one to that group, the residues form the non-prime side, and the prime
subsites hold no peptide residue at all and must be masked. An internally
quenched FRET substrate does not: its cut site is internal and unreported.

Dimension symbols: ``P`` subsites (8), ``A`` amino acids (20).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import NamedTuple

import numpy as np

from pep_compass.optimization.components.helpers.proteolysis.cleavage import (
    MEROPS_AMINO_ACIDS,
    MEROPS_SUBSITES,
)

# Index of P1 in the subsite order P4 P3 P2 P1 | P1' P2' P3' P4', and the number
# of subsites on the non-prime side.
P1_SUBSITE = MEROPS_SUBSITES.index("P1")
N_NON_PRIME_SUBSITES = P1_SUBSITE + 1

THREE_TO_ONE: dict[str, str] = {
    "ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C",
    "gln": "Q", "glu": "E", "gly": "G", "his": "H", "ile": "I",
    "leu": "L", "lys": "K", "met": "M", "phe": "F", "pro": "P",
    "ser": "S", "thr": "T", "trp": "W", "tyr": "Y", "val": "V",
}

# Stereochemistry and backbone-nitrogen prefixes that do not change the residue.
RESIDUE_PREFIXES = ("l-", "d-", "n-", "alpha-", "beta-", "n-alpha-", "n-omega-")

# Groups capping the N-terminus of a synthetic substrate. They block the
# non-prime context but leave the residues themselves interpretable. Matched as a
# prefix of the whole description, since these labels carry internal hyphens.
N_TERMINAL_BLOCK_PREFIXES: tuple[str, ...] = (
    "abz", "ac", "acetyl", "benzoyl", "bz", "boc", "cbz", "dabcyl", "dnp",
    "fmoc", "glt", "mca", "meosuc", "suc", "succinyl", "tos", "z",
    "n-benzoyl", "n-succinyl", "n-acetyl", "n-carbobenzoxy", "pyr", "mocac",
)

# Reporter groups released on hydrolysis. Their presence fixes the scissile bond.
# REMARK: Matched as a suffix of the whole description rather than as a token,
# because names such as "7-amido-4-carbamoylmethylcoumarin" are themselves
# hyphenated and would be split apart by token-wise matching.
C_TERMINAL_REPORTER_SUFFIXES: tuple[str, ...] = (
    "7-amido-4-carbamoylmethylcoumarin", "7-amido-4-methylcoumarin",
    "7-amino-4-methylcoumarin", "4-methoxy-2-naphthylamide", "2-naphthylamide",
    "4-nitroanilide", "p-nitroanilide", "nitroanilide", "4-nitrophenyl",
    "thiobenzyl", "eddnp", "ednp", "yno2", "pna", "amc", "afc", "mna", "acc",
    "smbz", "pnp",
)

# Donor/quencher pairs marking an internally quenched (FRET) substrate, whose cut
# site is internal and not reported by the database.
# REMARK: Searched anywhere in the description, not as a suffix: a quencher is
# frequently written parenthetically, as in
# "o-aminobenzoyl-GFSPFRN-(N-2,4-dinitrophenyl)".
FRET_DONORS: tuple[str, ...] = ("abz", "o-aminobenzoyl", "aminobenzoyl", "mca", "mocac")
FRET_QUENCHERS: tuple[str, ...] = ("eddnp", "ednp", "yno2", "dnp", "dinitrophenyl", "dabcyl")

ONE_LETTER_RUN = re.compile(rf"[{MEROPS_AMINO_ACIDS}]{{4,}}")

# Placeholders used by source databases in place of a substrate name. They are not
# macromolecules and must never be counted as an identified substrate.
PLACEHOLDER_SUBSTRATES: frozenset[str] = frozenset(
    {"additional information", "more", "?", "-", "n.a.", "na", "not available", "unknown"}
)


class SubstrateKind(str, Enum):
    """How a substrate description could be interpreted."""

    PEPTIDE_ONE_LETTER = "peptide_one_letter"
    PEPTIDE_THREE_LETTER = "peptide_three_letter"
    SINGLE_RESIDUE = "single_residue"
    MACROMOLECULE = "macromolecule"
    UNRESOLVED = "unresolved"


class CleavageSiteSource(str, Enum):
    """How the scissile bond of a substrate was determined."""

    REPORTER_GROUP = "reporter_group"
    UNKNOWN_INTERNAL = "unknown_internal"
    NOT_APPLICABLE = "not_applicable"


class SubstrateParse(NamedTuple):
    """Outcome of interpreting one substrate description.

    ``sequence`` holds the recovered residues in one-letter code, empty when
    nothing could be recovered. ``has_n_block`` and ``reporter`` record the
    synthetic groups found, because they determine which subsites can carry a
    peptide residue at all.
    """

    raw: str
    sequence: str
    kind: SubstrateKind
    has_n_block: bool
    reporter: str | None
    is_fret: bool


def _strip_residue_prefix(token: str) -> str:
    """Remove stereochemistry or backbone prefixes from a residue token.

    :param token: Lowercase token such as ``"l-ala"``.
    :return: Token without a leading configuration prefix.
    """

    changed = True
    while changed:
        changed = False
        for prefix in RESIDUE_PREFIXES:
            if token.startswith(prefix) and len(token) > len(prefix):
                token = token[len(prefix) :]
                changed = True
    return token


def parse_substrate(text: str | None) -> SubstrateParse:
    """Interpret a free-text substrate description.

    The one-letter form is tried first, because a run of standard uppercase
    residues inside a labelled substrate is unambiguous. The three-letter form is
    tried next, token by token. Anything else stays unresolved rather than being
    coerced into a sequence.

    :param text: Substrate description as reported by the source database.
    :return: The parse outcome, with an empty sequence when nothing was resolved.
    """

    raw = (text or "").strip()
    if not raw:
        return SubstrateParse(raw, "", SubstrateKind.UNRESOLVED, False, None, False)

    lowered_raw = raw.lower()
    if lowered_raw in PLACEHOLDER_SUBSTRATES:
        return SubstrateParse(raw, "", SubstrateKind.UNRESOLVED, False, None, False)

    has_n_block = any(
        lowered_raw.startswith(f"{prefix}-") for prefix in N_TERMINAL_BLOCK_PREFIXES
    )
    reporter = next(
        (suffix for suffix in C_TERMINAL_REPORTER_SUFFIXES if lowered_raw.endswith(suffix)),
        None,
    )
    is_fret = any(donor in lowered_raw for donor in FRET_DONORS) and any(
        quencher in lowered_raw for quencher in FRET_QUENCHERS
    )
    lowered = [token for token in re.split(r"[-\s]+", lowered_raw) if token]

    ## One-letter run inside a labelled synthetic substrate
    runs = ONE_LETTER_RUN.findall(raw)
    if runs:
        longest = max(runs, key=len)
        return SubstrateParse(
            raw, longest, SubstrateKind.PEPTIDE_ONE_LETTER, has_n_block, reporter, is_fret
        )

    ## Three-letter residue tokens joined by hyphens
    residues: list[str] = []
    for token in lowered:
        residue = THREE_TO_ONE.get(_strip_residue_prefix(token))
        if residue is not None:
            residues.append(residue)
    if residues:
        kind = SubstrateKind.SINGLE_RESIDUE if len(residues) == 1 else SubstrateKind.PEPTIDE_THREE_LETTER
        return SubstrateParse(raw, "".join(residues), kind, has_n_block, reporter, is_fret)

    ## Nothing residue-like: a protein, a polymer or an unparsable label
    is_plain_name = re.fullmatch(r"[A-Za-z][A-Za-z \-]*", raw) is not None
    kind = SubstrateKind.MACROMOLECULE if is_plain_name else SubstrateKind.UNRESOLVED
    return SubstrateParse(raw, "", kind, has_n_block, reporter, is_fret)


def infer_cleavage_bond(parse: SubstrateParse) -> tuple[int | None, CleavageSiteSource]:
    """Infer which bond of a parsed substrate was hydrolysed.

    A reporter group leaves from the C-terminus, so the scissile bond follows the
    last residue. That bond is reported as ``len(sequence) - 1``: the residues
    occupy the non-prime subsites and the prime side holds the reporter, not a
    peptide residue.

    :param parse: Outcome of :func:`parse_substrate`.
    :return: Tuple of bond index (``None`` when unresolved) and how it was
        determined.
    """

    if not parse.sequence:
        return None, CleavageSiteSource.NOT_APPLICABLE
    if parse.is_fret:
        return None, CleavageSiteSource.UNKNOWN_INTERNAL
    if parse.reporter is not None:
        return len(parse.sequence) - 1, CleavageSiteSource.REPORTER_GROUP
    return None, CleavageSiteSource.UNKNOWN_INTERNAL


def reporter_window_indices(parse: SubstrateParse) -> np.ndarray:
    """Return the residue index of each subsite for a reporter substrate.

    The scissile bond of a chromogenic or fluorogenic substrate is the amide to
    the leaving group, which lies outside the peptide. The residues therefore
    occupy the non-prime subsites, with :math:`P_1` on the last residue, and the
    prime subsites hold the reporter rather than a residue.

    This convention cannot be expressed as an internal peptide bond, which is why
    it has its own index function rather than reusing
    :func:`~pep_compass.optimization.components.helpers.proteolysis.geometry.window_residue_indices`.

    :param parse: Outcome of :func:`parse_substrate`.
    :return: Residue index per subsite, ``-1`` where the subsite has no residue,
        shape ``(P,)``.
    """

    indices = np.full(len(MEROPS_SUBSITES), -1, dtype=int)  # (P,)
    bond, source = infer_cleavage_bond(parse)
    if bond is None or source is not CleavageSiteSource.REPORTER_GROUP:
        return indices

    ## P1 is the last residue, P2 the one before it, and so on back to P4.
    for offset in range(N_NON_PRIME_SUBSITES):
        residue_index = len(parse.sequence) - 1 - offset
        if residue_index < 0:
            break
        indices[P1_SUBSITE - offset] = residue_index
    return indices


def resolvable_subsites(parse: SubstrateParse) -> int:
    """Count the subsites that a resolved substrate can actually populate.

    For a reporter substrate the prime side is chemistry, not peptide, so at most
    the four non-prime subsites can be filled, and fewer when the peptide is
    short. An unresolved cut site populates nothing.

    :param parse: Outcome of :func:`parse_substrate`.
    :return: Number of subsites carrying a genuine peptide residue, ``0`` when the
        window cannot be built.
    """

    bond, source = infer_cleavage_bond(parse)
    if bond is None or source is not CleavageSiteSource.REPORTER_GROUP:
        return 0
    ## P1 sits on the last residue, so the non-prime side reaches back at most
    ## four residues and the prime side is entirely occupied by the reporter.
    return min(len(parse.sequence), 4)
