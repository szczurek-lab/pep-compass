"""Source-supported peptide chemistry and cut recovery shared by dataset parsers.

Migrated from the reviewed kinetics coverage notebook; no model-based site choice.
"""

import math
import re
from dataclasses import dataclass
from functools import cache
from itertools import permutations

INCLUDE_NONSTANDARD = True
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_NAMES = {
    "A": ("Ala", "alanine", "alanyl"),
    "R": ("Arg", "arginine", "arginyl"),
    "N": ("Asn", "asparagine", "asparaginyl"),
    "D": ("Asp", "aspartate", "aspartyl"),
    "C": ("Cys", "cysteine", "cysteinyl"),
    "Q": ("Gln", "glutamine", "glutaminyl"),
    "E": ("Glu", "glutamate", "glutamyl"),
    "G": ("Gly", "glycine", "glycyl"),
    "H": ("His", "histidine", "histidyl"),
    "I": ("Ile", "isoleucine", "isoleucyl"),
    "L": ("Leu", "leucine", "leucyl"),
    "K": ("Lys", "lysine", "lysyl"),
    "M": ("Met", "methionine", "methionyl"),
    "F": ("Phe", "phenylalanine", "phenylalanyl"),
    "P": ("Pro", "proline", "prolyl"),
    "S": ("Ser", "serine", "seryl"),
    "T": ("Thr", "threonine", "threonyl"),
    "W": ("Trp", "tryptophan", "tryptophyl"),
    "Y": ("Tyr", "tyrosine", "tyrosyl"),
    "V": ("Val", "valine", "valyl"),
}
RESIDUES = {
    name.lower(): symbol for symbol, names in AA_NAMES.items() for name in names
}
RESIDUES.update(
    {
        "nle": "Nle",
        "norleucine": "Nle",
        "nva": "Nva",
        "norvaline": "Nva",
        "aib": "Aib",
        "abu": "Abu",
        "orn": "Orn",
        "ornithine": "Orn",
        "4-nitrophenylalanine": "Phe(4-nitro)",
        "(4-nitrophenylalanine)": "Phe(4-nitro)",
    }
)
REPORTERS = (
    "7-amido-4-carbamoylmethylcoumarin",
    "7-amido-4-fluoromethylcoumarin",
    "7-amino-4-methylcoumarin",
    "7-amido-4-methylcoumarin",
    "4-methylcoumarin-7-amide",
    "4-methyl-coumaryl-7-amide",
    "4-methoxy-2-naphthylamide",
    "2-naphthylamide",
    "p-nitroanilide",
    "4-nitroanilide",
    "nitroanilide",
    "pna",
    "amc",
    "afc",
    "acc",
    "mna",
)
N_CAPS = (
    "n-alpha-benzoyl",
    "n-benzyloxycarbonyl",
    "n-carbobenzoxy",
    "n-succinyl",
    "n-acetyl",
    "n-benzoyl",
    "t-butyloxycarbonyl",
    "tert-butyloxycarbonyl",
    "benzyloxycarbonyl",
    "(o-aminobenzoyl)",
    "o-aminobenzoyl",
    "2-aminobenzoyl",
    "aminobenzoyl",
    "succinyl",
    "acetyl",
    "benzoyl",
    "glutaryl",
    "dabcyl",
    "meosuc",
    "mocac",
    "fmoc",
    "abz",
    "cbz",
    "boc",
    "dnp",
    "mca",
    "suc",
    "glt",
    "tos",
    "ac",
    "bz",
    "z",
    "h",
)
C_CAPS = (
    *REPORTERS,
    "(N-(2,4-dinitrophenyl)ethylenediamine)",
    "N-(2,4-dinitrophenyl)ethylenediamine",
    "eddnp",
    "ednp",
    "edans",
    "dabcyl",
    "dnp",
    "nh2",
    "oh",
)
RESIDUE_PATTERN = re.compile(
    "|".join(re.escape(name) for name in sorted(RESIDUES, key=len, reverse=True)),
    re.IGNORECASE,
)
UNIPROT = re.compile(
    r"(?<![A-Z0-9])(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?(?![A-Z0-9])"
)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
POINT = re.compile(rf"({NUMBER})(?:\s*(?:±|\+/-)\s*({NUMBER}))?")
BOND_MOTIF = re.compile(
    r"\bcleavage\s+of\s+(?:the\s+)?([ACDEFGHIKLMNPQRSTVWYX])-([ACDEFGHIKLMNPQRSTVWYX])\s+bond\b"
)


@dataclass(frozen=True)
class Peptide:
    """Parsed source chemistry; unknown residue spellings are rejected."""

    residues: tuple
    n_cap: str | None
    c_cap: str | None
    reported_cut: int | None = None
    raw_label: str = ""


def accession_ids(values):
    """Extract reported UniProt accessions, including oligomer annotations.

    :param values: Original identifier strings or a list of strings.
    :return: Unique accessions; no identity is invented for missing identifiers.
    :rtype: frozenset
    """
    values = values if isinstance(values, list) else [values]
    return frozenset(
        hit for value in values if value for hit in UNIPROT.findall(str(value))
    )


def point_value(text):
    """Recognize a nonnegative finite point, keeping intervals unresolved.

    :param text: Original scalar or numeric text.
    :return: Whether the value is a supported numeric point.
    :rtype: bool
    """
    match = POINT.fullmatch(str(text).strip())
    return bool(match) and all(
        math.isfinite(float(x)) and float(x) >= 0
        for x in match.groups()
        if x is not None
    )


def residues_from_text(text):
    """Parse the whole sequence spelling, without skipping unknown chemistry.

    :param str text: Sequence portion after recognized terminal labels.
    :return: Residue tokens, or null for unsupported/ambiguous syntax.
    :rtype: tuple | None
    """
    if re.fullmatch(rf"[{AA}]+", text) and (
        len(text) >= 4 or text.lower() not in RESIDUES
    ):
        return tuple(text)
    if re.fullmatch(rf"[{AA}]+", text) and text.lower() in RESIDUES:
        return None  # Uppercase triplets can denote one residue or a short peptide.
    tokens, position = [], 0
    while position < len(text):
        stereo = re.match(r"([DL])-", text[position:])
        if stereo:
            position += stereo.end()
        match = RESIDUE_PATTERN.match(text, position)
        if match is None:
            return None
        token = RESIDUES[match.group().lower()]
        if stereo and stereo[1] == "D":
            token = "D-" + token
        if not INCLUDE_NONSTANDARD and token not in AA:
            return None
        tokens.append(token)
        position = match.end()
        if position == len(text):
            break
        separator = re.match(r"[-\s]+", text[position:])
        if separator is None:
            return None
        position += separator.end()
    return tuple(tokens) if tokens else None


@cache
def parse_peptide(raw):
    """Parse a complete label, preserving recognized terminal groups and cuts.

    :param str raw: Original species name, comment or BRENDA substrate label.
    :return: Parsed source chemistry, or null if it cannot be resolved safely.
    :rtype: Peptide | None
    """
    body, n_cap, c_cap = raw.strip(), None, None
    for cap in sorted(N_CAPS, key=len, reverse=True):
        match = re.match(re.escape(cap) + r"[-\s]+", body, re.IGNORECASE)
        if match:
            n_cap, body = (None if cap == "h" else cap.lower()), body[match.end() :]
            break
    for cap in sorted(C_CAPS, key=len, reverse=True):
        match = re.search(r"[-\s]+" + re.escape(cap) + r"$", body, re.IGNORECASE)
        if match:
            c_cap, body = (None if cap == "oh" else cap.lower()), body[: match.start()]
            break
    cut = None
    if body.count("/") == 1:
        left, right = body.split("/")
        if not re.fullmatch(rf"[{AA}]+", left) or not re.fullmatch(rf"[{AA}]+", right):
            return None
        tokens, cut = tuple(left + right), len(left)
    else:
        tokens = residues_from_text(body)
    return Peptide(tokens, n_cap, c_cap, cut, raw) if tokens else None


def species_peptide(species):
    """Require source name and comment to agree when both yield sequences.

    :param dict species: Original reaction participant.
    :return: One unambiguous parsed descriptor or null.
    :rtype: Peptide | None
    """
    descriptions = [(species.get("compound") or {}).get("name"), species.get("comment")]
    parsed = [
        p for text in descriptions if text and (p := parse_peptide(text)) is not None
    ]
    if not parsed or len({p.residues for p in parsed}) != 1:
        return None
    return next((p for p in parsed if p.reported_cut is not None), parsed[0])


def resolved_sites(peptide, comment="", products=()):
    """Resolve cuts only from source notation, unique bond matches or products.

    :param Peptide peptide: Parsed substrate.
    :param str comment: Substrate-associated source comment.
    :param products: Parsed product descriptors from the same source reaction.
    :return: Position/evidence pairs; ambiguous candidate positions are omitted.
    :rtype: list
    """
    sites = {}
    if peptide.reported_cut is not None:
        sites[peptide.reported_cut] = "reported_separator"
    for left, right in BOND_MOTIF.findall(comment):
        positions = [
            i
            for i in range(1, len(peptide.residues))
            if (left == "X" or peptide.residues[i - 1] == left)
            and (right == "X" or peptide.residues[i] == right)
        ]
        if len(positions) == 1:
            sites.setdefault(positions[0], "reported_bond_unique_match")
    candidates = set()
    if len(products) == 2:
        for left, right in permutations(products):
            if (
                left.residues + right.residues == peptide.residues
                and left.n_cap == peptide.n_cap
                and right.c_cap == peptide.c_cap
                and left.c_cap is None
                and right.n_cap is None
            ):
                candidates.add(len(left.residues))
    if len(candidates) == 1:
        sites.setdefault(next(iter(candidates)), "reported_products_unique_match")
    if not sites and peptide.c_cap in {cap.lower() for cap in REPORTERS}:
        sites.setdefault(len(peptide.residues), "reporter_linkage")
    return sorted(sites.items())
