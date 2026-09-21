r"""Event geometry of proteolytic cleavage: which bonds a peptidase may cut at all.

``cleavage.py`` scores every internal bond of a peptide against every protease in
the panel. That is only correct for endopeptidases. An exopeptidase attacks a
terminus and nothing else, so an internal bond is not a weak cleavage site for it
-- it is a **forbidden event**, and it must never be scored or encoded as an
observed non-cleavage.

This module separates three things that are easy to conflate:

* the **geometry** of a peptidase (:class:`EventGeometry`), derived from its EC
  sub-subclass and, failing that, from its MEROPS name;
* the **allowed bonds** of a peptide under that geometry
  (:func:`allowed_cut_positions`);
* the **observable subsites** of a given bond (:func:`subsite_observability_mask`),
  since a bond near a terminus simply has no :math:`P_4` or :math:`P_4'` residue.

The third point matters for scoring: ``CleavagePotential`` substitutes a
background residue distribution for subsites that fall outside the peptide, which
makes a terminal bond look like an ordinary one. A masked window score
(:func:`masked_window_score`) sums only the subsites that exist, which is the
convention the calibration specification requires for partial windows.

Dimension symbols: ``L`` residues, ``P`` subsites (8), ``A`` amino acids (20).
"""

from __future__ import annotations

import re
from enum import Enum

from collections.abc import Sequence

import numpy as np

from pep_compass.optimization.components.helpers.proteolysis.cleavage import (
    MEROPS_AMINO_ACIDS,
    MEROPS_SUBSITE_OFFSETS,
    MEROPS_SUBSITES,
)


class EventGeometry(str, Enum):
    """Kinds of cleavage event a peptidase can produce.

    The members distinguish where a peptidase may cut, not how fast it cuts.
    ``NON_HYDROLASE`` marks entries that do not hydrolyse peptide bonds at all
    and must be excluded from the hydrolysis model; ``UNKNOWN`` marks entries
    whose geometry could not be resolved and which therefore must not be scored
    silently.
    """

    ENDOPEPTIDASE = "endopeptidase"
    AMINOPEPTIDASE = "aminopeptidase"
    DIPEPTIDYL_PEPTIDASE = "dipeptidyl_peptidase"
    TRIPEPTIDYL_PEPTIDASE = "tripeptidyl_peptidase"
    CARBOXYPEPTIDASE = "carboxypeptidase"
    PEPTIDYL_DIPEPTIDASE = "peptidyl_dipeptidase"
    OMEGA_PEPTIDASE = "omega_peptidase"
    DIPEPTIDASE = "dipeptidase"
    NON_HYDROLASE = "non_hydrolase"
    UNKNOWN = "unknown"


# Exact EC numbers whose geometry differs from their sub-subclass.
# REMARK: Sub-subclass 3.4.14 is "dipeptidyl-peptidases and tripeptidyl-peptidases"
# and therefore mixes two geometries. The members that release a tripeptide are
# listed here, because the difference is a different state transition: three
# residues leave the peptide instead of two.
EC_NUMBER_GEOMETRY: dict[str, EventGeometry] = {
    "3.4.14.9": EventGeometry.TRIPEPTIDYL_PEPTIDASE,   # tripeptidyl-peptidase I
    "3.4.14.10": EventGeometry.TRIPEPTIDYL_PEPTIDASE,  # tripeptidyl-peptidase II
}

# EC sub-subclass -> geometry. This is the authoritative mapping: the peptidase
# section of the EC hierarchy is organized precisely by attack geometry.
EC_SUBSUBCLASS_GEOMETRY: dict[str, EventGeometry] = {
    "3.4.11": EventGeometry.AMINOPEPTIDASE,
    "3.4.13": EventGeometry.DIPEPTIDASE,
    "3.4.14": EventGeometry.DIPEPTIDYL_PEPTIDASE,
    "3.4.15": EventGeometry.PEPTIDYL_DIPEPTIDASE,
    "3.4.16": EventGeometry.CARBOXYPEPTIDASE,
    "3.4.17": EventGeometry.CARBOXYPEPTIDASE,
    "3.4.18": EventGeometry.CARBOXYPEPTIDASE,
    "3.4.19": EventGeometry.OMEGA_PEPTIDASE,
    "3.4.21": EventGeometry.ENDOPEPTIDASE,
    "3.4.22": EventGeometry.ENDOPEPTIDASE,
    "3.4.23": EventGeometry.ENDOPEPTIDASE,
    "3.4.24": EventGeometry.ENDOPEPTIDASE,
    "3.4.25": EventGeometry.ENDOPEPTIDASE,
    "3.4.99": EventGeometry.UNKNOWN,
}

# Name fragments used only when no EC number is available. Order matters: the
# more specific pattern must be tested first ("tripeptidyl-peptidase" contains
# "peptidyl-peptidase").
NAME_GEOMETRY_RULES: tuple[tuple[str, EventGeometry], ...] = (
    (r"tripeptidyl[- ]?peptidase", EventGeometry.TRIPEPTIDYL_PEPTIDASE),
    (r"dipeptidyl[- ]?(amino)?peptidase", EventGeometry.DIPEPTIDYL_PEPTIDASE),
    (r"peptidyl[- ]?dipeptidase", EventGeometry.PEPTIDYL_DIPEPTIDASE),
    (r"carboxypeptidase", EventGeometry.CARBOXYPEPTIDASE),
    (r"aminopeptidase", EventGeometry.AMINOPEPTIDASE),
    (r"\bdipeptidase\b", EventGeometry.DIPEPTIDASE),
    (r"endopeptidase|elastase|granzyme|chymase|tryptase|thrombin|plasmin|kallikrein"
     r"|metallopeptidase|metalloproteinase|cathepsin g|coagulation factor|complement"
     r"|activator|convertase|neprilysin|adamts", EventGeometry.ENDOPEPTIDASE),
)

# Geometries whose event is a terminal release rather than an internal cut, with
# the number of residues removed from the N- and C-terminus respectively.
TERMINAL_RELEASE: dict[EventGeometry, tuple[int, int]] = {
    EventGeometry.AMINOPEPTIDASE: (1, 0),
    EventGeometry.DIPEPTIDYL_PEPTIDASE: (2, 0),
    EventGeometry.TRIPEPTIDYL_PEPTIDASE: (3, 0),
    EventGeometry.CARBOXYPEPTIDASE: (0, 1),
    EventGeometry.PEPTIDYL_DIPEPTIDASE: (0, 2),
}


def geometry_from_ec(ec_number: str | None) -> EventGeometry:
    """Derive the event geometry from an EC number.

    An exact EC number listed in :data:`EC_NUMBER_GEOMETRY` overrides its
    sub-subclass, which is needed because ``3.4.14`` holds both dipeptidyl- and
    tripeptidyl-peptidases.

    :param ec_number: Full EC number such as ``"3.4.21.5"``. Empty or malformed
        values resolve to :attr:`EventGeometry.UNKNOWN`.
    :return: The geometry implied by the EC number.
    """

    if not ec_number:
        return EventGeometry.UNKNOWN
    normalized = str(ec_number).strip()
    ## An exact match wins over the sub-subclass, because a sub-subclass can
    ## contain more than one geometry.
    if normalized in EC_NUMBER_GEOMETRY:
        return EC_NUMBER_GEOMETRY[normalized]
    parts = normalized.split(".")
    if len(parts) < 3:
        return EventGeometry.UNKNOWN
    return EC_SUBSUBCLASS_GEOMETRY.get(".".join(parts[:3]), EventGeometry.UNKNOWN)


def geometry_from_name(name: str | None) -> EventGeometry:
    """Derive the event geometry from a MEROPS display name.

    This is the fallback for panel entries whose UniProt record carries no EC
    number. It is deliberately conservative: an unmatched name stays
    :attr:`EventGeometry.UNKNOWN` rather than defaulting to an endopeptidase.

    :param name: MEROPS peptidase name.
    :return: The geometry implied by the name, or ``UNKNOWN``.
    """

    if not name:
        return EventGeometry.UNKNOWN
    lowered = str(name).lower()
    for pattern, geometry in NAME_GEOMETRY_RULES:
        if re.search(pattern, lowered):
            return geometry
    return EventGeometry.UNKNOWN


def resolve_geometry(
    ec_number: str | None,
    name: str | None = None,
    include_in_hydrolysis_model: bool = True,
) -> tuple[EventGeometry, str]:
    """Resolve the geometry of one panel entry and record how it was resolved.

    :param ec_number: EC number, when the UniProt mapping provided one.
    :param name: MEROPS display name, used as a fallback.
    :param include_in_hydrolysis_model: ``False`` for entries excluded from the
        hydrolysis model, which resolve to :attr:`EventGeometry.NON_HYDROLASE`.
    :return: Tuple of geometry and provenance label (``"ec"``, ``"name"``,
        ``"manifest_exclusion"`` or ``"unresolved"``).
    """

    if not include_in_hydrolysis_model:
        return EventGeometry.NON_HYDROLASE, "manifest_exclusion"
    geometry = geometry_from_ec(ec_number)
    if geometry is not EventGeometry.UNKNOWN:
        return geometry, "ec"
    geometry = geometry_from_name(name)
    if geometry is not EventGeometry.UNKNOWN:
        return geometry, "name"
    return EventGeometry.UNKNOWN, "unresolved"


def allowed_cut_positions(length: int, geometry: EventGeometry) -> np.ndarray:
    """Return the bonds a peptidase of this geometry may cut in a peptide.

    Bond ``b`` sits between residues ``b`` and ``b + 1`` (zero-based), so a
    peptide of ``L`` residues has ``L - 1`` bonds.

    :param length: Peptide length ``L`` in residues.
    :param geometry: Event geometry of the peptidase.
    :return: Boolean mask over bonds, shape ``(L - 1,)``.
    :raises ValueError: If ``length`` is negative.
    """

    if length < 0:
        raise ValueError("length must be non-negative.")
    n_bonds = max(length - 1, 0)
    mask = np.zeros(n_bonds, dtype=bool)  # (L - 1,)
    if n_bonds == 0:
        return mask

    if geometry is EventGeometry.ENDOPEPTIDASE:
        mask[:] = True
        return mask

    ## Terminal releases occupy exactly one bond, and only if the peptide is
    ## long enough to leave a remaining fragment on the other side.
    if geometry in TERMINAL_RELEASE:
        from_n, from_c = TERMINAL_RELEASE[geometry]
        if from_n:
            if length > from_n:
                mask[from_n - 1] = True
        else:
            if length > from_c:
                mask[n_bonds - from_c] = True
        return mask

    ## A dipeptidase acts only on a free dipeptide; every other geometry is
    ## either non-hydrolytic or unresolved and must not be scored.
    if geometry is EventGeometry.DIPEPTIDASE and length == 2:
        mask[0] = True
    return mask


def subsite_observability_mask(length: int, bond: int) -> np.ndarray:
    """Return which subsites of one bond lie inside the peptide.

    :param length: Peptide length ``L`` in residues.
    :param bond: Zero-based bond index, between residues ``bond`` and ``bond + 1``.
    :return: Boolean mask over ``P4 … P4'``, shape ``(P,)``.
    :raises ValueError: If ``bond`` is not a bond of a peptide of this length.
    """

    if not 0 <= bond < max(length - 1, 0):
        raise ValueError(f"bond {bond} is not a bond of a peptide of length {length}.")
    positions = bond + np.asarray(MEROPS_SUBSITE_OFFSETS)  # (P,)
    return (positions >= 0) & (positions < length)  # (P,)


def window_residue_indices(length: int, bond: int) -> np.ndarray:
    """Return the residue index of each subsite of one bond.

    Subsites outside the peptide are reported as ``-1`` so callers cannot read
    them by accident.

    :param length: Peptide length ``L`` in residues.
    :param bond: Zero-based bond index.
    :return: Residue indices per subsite, shape ``(P,)``.
    """

    positions = bond + np.asarray(MEROPS_SUBSITE_OFFSETS)  # (P,)
    return np.where(subsite_observability_mask(length, bond), positions, -1)  # (P,)


def enumerate_window_scores(
    sequence: str,
    matrix: np.ndarray,
    geometry: EventGeometry = EventGeometry.ENDOPEPTIDASE,
    background: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Score every bond of a peptide that the geometry admits.

    A substrate whose cleavage site was not reported still has a well-defined set
    of candidate bonds. Scoring all of them turns the unknown site into a latent
    variable instead of a value that has to be invented.

    :param sequence: Peptide sequence over the MEROPS alphabet.
    :param matrix: Specificity log-probabilities of one protease, shape ``(P, A)``.
    :param geometry: Event geometry restricting which bonds are admissible.
    :param background: Background residue distribution ``(A,)``, uniform by
        default.
    :return: Tuple of per-bond scores ``(L - 1,)``, the number of observed
        subsites per bond ``(L - 1,)`` and the admissibility mask ``(L - 1,)``.
        Inadmissible bonds carry ``-inf`` so that they cannot win a maximum.
    """

    length = len(sequence)
    admissible = allowed_cut_positions(length, geometry)  # (L - 1,)
    scores = np.full(admissible.shape, -np.inf)  # (L - 1,)
    observed = np.zeros(admissible.shape, dtype=int)  # (L - 1,)

    for bond in np.flatnonzero(admissible):
        scores[bond], observed[bond] = masked_window_score(
            sequence, int(bond), matrix, background=background
        )
    return scores, observed, admissible


def implied_cleavage_bond(
    sequence: str,
    matrix: np.ndarray,
    geometry: EventGeometry = EventGeometry.ENDOPEPTIDASE,
    background: np.ndarray | None = None,
    min_observed_subsites: int = 1,
) -> tuple[int | None, float, float]:
    r"""Return the bond a specificity matrix ranks highest, and how clearly.

    The returned bond is **implied by the matrix**, not measured. It is a
    hypothesis about where the enzyme cut, and it must never be recorded as an
    observed cleavage site. The margin is reported so that a substrate with two
    near-equal candidates can be treated differently from one with a clear
    winner.

    :param sequence: Peptide sequence over the MEROPS alphabet.
    :param matrix: Specificity log-probabilities of one protease, shape ``(P, A)``.
    :param geometry: Event geometry restricting which bonds are admissible.
    :param background: Background residue distribution ``(A,)``, uniform by
        default.
    :param min_observed_subsites: Bonds populating fewer subsites than this are
        not considered, which excludes windows too truncated to compare.
    :return: Tuple of the implied bond (``None`` when no bond qualifies), its
        score, and the margin to the next best bond (``inf`` when it is the only
        candidate).
    """

    scores, observed, admissible = enumerate_window_scores(sequence, matrix, geometry, background)
    eligible = admissible & (observed >= min_observed_subsites)
    if not eligible.any():
        return None, float("-inf"), float("inf")

    candidate_scores = np.where(eligible, scores, -np.inf)  # (L - 1,)
    best = int(np.argmax(candidate_scores))
    ordered = np.sort(candidate_scores[eligible])[::-1]
    margin = float(ordered[0] - ordered[1]) if ordered.size > 1 else float("inf")
    return best, float(candidate_scores[best]), margin


def restrict_window_indices(
    residue_indices: np.ndarray, subsites: Sequence[str]
) -> np.ndarray:
    """Blank the subsites outside a chosen subset, leaving the rest in place.

    A window score is a sum over the subsites that carry a residue, so removing a
    subsite from the sum is done by marking it absent rather than by shortening
    the array. This keeps the subsite axis aligned with the specificity matrix,
    which is what allows a restricted window to be scored by the same function as
    a full one.

    :param residue_indices: Residue index per subsite, ``-1`` where the subsite
        has no residue, shape ``(P,)``.
    :param subsites: Names of the subsites to keep, from
        :data:`MEROPS_SUBSITES`.
    :return: A copy with every other subsite set to ``-1``, shape ``(P,)``.
    :raises ValueError: If a name is not a known subsite.
    """

    unknown = set(subsites) - set(MEROPS_SUBSITES)
    if unknown:
        raise ValueError(f"Unknown subsites: {sorted(unknown)}.")

    keep = np.array([name in set(subsites) for name in MEROPS_SUBSITES])  # (P,)
    restricted = np.array(residue_indices, dtype=int).copy()  # (P,)
    restricted[~keep[: len(restricted)]] = -1
    return restricted


def masked_score_from_indices(
    sequence: str,
    residue_indices: np.ndarray,
    matrix: np.ndarray,
    background: np.ndarray | None = None,
) -> tuple[float, int]:
    r"""Score one window given the residue index of each subsite.

    This is the scoring step shared by every window convention. The caller
    decides which residue each subsite sees and marks absent subsites with
    ``-1``; an absent subsite contributes nothing, which is what distinguishes a
    masked score from one that substitutes a background residue.

    :param sequence: Peptide sequence over the MEROPS alphabet.
    :param residue_indices: Residue index per subsite, ``-1`` where the subsite
        has no residue, shape ``(P,)``.
    :param matrix: Specificity log-probabilities of one protease, shape ``(P, A)``.
    :param background: Background residue distribution ``(A,)``, uniform by
        default.
    :return: Tuple of the masked score and the number of observed subsites.
    :raises ValueError: If the sequence holds a residue outside the MEROPS
        alphabet.
    """

    residue_column = {residue: column for column, residue in enumerate(MEROPS_AMINO_ACIDS)}
    if background is None:
        background = np.full(matrix.shape[-1], 1.0 / matrix.shape[-1])
    log_background = np.log(np.asarray(background, dtype=np.float64))  # (A,)

    score = 0.0
    observed = 0
    for subsite, residue_index in enumerate(residue_indices):
        if residue_index < 0:
            continue
        residue = sequence[residue_index]
        if residue not in residue_column:
            raise ValueError(f"residue {residue!r} is outside the MEROPS alphabet.")
        column = residue_column[residue]
        score += float(matrix[subsite, column] - log_background[column])
        observed += 1
    return score, observed


def masked_window_score(
    sequence: str,
    bond: int,
    matrix: np.ndarray,
    background: np.ndarray | None = None,
) -> tuple[float, int]:
    r"""Score one bond over the subsites that actually exist.

    This is the masked window score of the calibration specification,
    :math:`S^{M,\mathrm{masked}}_{\pi,b} = \sum_{p:\,m_p=1} M_{\pi,p,a_p}`,
    expressed as log-odds against a background so that dropping a subsite is
    neutral rather than a free bonus.

    :param sequence: Peptide sequence over the MEROPS alphabet.
    :param bond: Zero-based bond index.
    :param matrix: Specificity log-probabilities of one protease, shape ``(P, A)``.
    :param background: Background residue distribution ``(A,)``, uniform by
        default.
    :return: Tuple of the masked score and the number of observed subsites.
    :raises ValueError: If the sequence holds a residue outside the MEROPS
        alphabet.
    """

    return masked_score_from_indices(
        sequence, window_residue_indices(len(sequence), bond), matrix, background
    )
