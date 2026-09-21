"""Comparing an independent cleavage dataset against a MEROPS specificity matrix.

A MEROPS matrix is a tally of cleavages reported across the literature. Several
public datasets profile one protease exhaustively in a single experiment, and
those are the only external check on a matrix that exists. To be a check rather
than a second opinion of unknown weight, every such dataset has to be reduced to
the same object as the matrix and compared on the same axes.

This module provides that reduction and that comparison, so that two datasets
measured by different methods can be placed side by side.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from pep_compass.optimization.components.helpers.proteolysis.cleavage import (
    MEROPS_AMINO_ACIDS,
    MEROPS_SUBSITES,
)
from pep_compass.optimization.components.helpers.proteolysis.specificity import (
    MEROPS_SMOOTHING_ALPHA,
)


#: Characters a source may use for a subsite that was not observed.
GAP_CHARACTERS = frozenset({"-", ".", "_", "X", " "})

_RESIDUE_INDEX = {residue: index for index, residue in enumerate(MEROPS_AMINO_ACIDS)}


def specificity_counts_from_windows(
    windows: Iterable[str], n_subsites: int = len(MEROPS_SUBSITES)
) -> np.ndarray:
    """Tally residue occurrences per subsite over a set of cleavage windows.

    This is the same reduction MEROPS performs on its own records: one cleavage
    contributes one count at each subsite it resolves. Positions carrying a gap
    character contribute nothing, so a source that resolves six of eight subsites
    yields a matrix with two empty rows rather than two invented ones.

    :param windows: Cleavage windows, each a string of ``n_subsites`` characters
        in the order ``P4 … P4'``.
    :param n_subsites: Expected window length.
    :return: Integer counts, shape ``(n_subsites, 20)``.
    :raises ValueError: If a window has the wrong length.
    """

    counts = np.zeros((n_subsites, len(MEROPS_AMINO_ACIDS)), dtype=np.int64)  # (P, A)
    for window in windows:
        if len(window) != n_subsites:
            raise ValueError(f"Window {window!r} has length {len(window)}, expected {n_subsites}.")
        for position, residue in enumerate(window):
            if residue in GAP_CHARACTERS:
                continue
            index = _RESIDUE_INDEX.get(residue)
            if index is not None:
                counts[position, index] += 1
    return counts


def log_probability_matrix(
    counts: np.ndarray, alpha: float = MEROPS_SMOOTHING_ALPHA
) -> np.ndarray:
    """Convert subsite counts into smoothed log-probabilities.

    The prior matches the one behind the distributed MEROPS matrices, so a matrix
    built here and a matrix read from MEROPS are on the same footing and their
    difference is a difference in evidence rather than in convention.

    :param counts: Subsite counts, shape ``(P, A)``.
    :param alpha: Dirichlet prior added to every count.
    :return: Log-probabilities, shape ``(P, A)``.
    :raises ValueError: If ``alpha`` is not positive.
    """

    if alpha <= 0.0:
        raise ValueError("alpha must be strictly positive.")

    smoothed = counts.astype(np.float64) + alpha  # (P, A)
    return np.log(smoothed / smoothed.sum(axis=1, keepdims=True))  # (P, A)


def _jensen_shannon_bits(first: np.ndarray, second: np.ndarray) -> float:
    """Return the Jensen-Shannon divergence between two distributions, in bits.

    :param first: Probability vector, shape ``(A,)``.
    :param second: Probability vector, shape ``(A,)``.
    :return: Divergence in bits, between 0 and 1.
    """

    mixture = 0.5 * (first + second)  # (A,)

    def relative_entropy(p: np.ndarray, q: np.ndarray) -> float:
        support = p > 0
        return float(np.sum(p[support] * np.log2(p[support] / q[support])))

    return 0.5 * relative_entropy(first, mixture) + 0.5 * relative_entropy(second, mixture)


def compare_specificity_matrices(
    reference: np.ndarray,
    candidate: np.ndarray,
    observed_subsites: Sequence[bool] | None = None,
) -> list[dict[str, float | str | bool]]:
    """Compare two specificity matrices subsite by subsite.

    Three quantities are reported per subsite because they fail differently. The
    rank correlation asks whether the two sources order the residues the same
    way; the Jensen-Shannon divergence asks how far apart the distributions are
    regardless of order; the top-residue agreement asks the single question a
    reader of a sequence logo would ask.

    :param reference: Log-probabilities of the reference matrix, shape ``(P, A)``.
    :param candidate: Log-probabilities of the candidate matrix, same shape.
    :param observed_subsites: Per-subsite flag marking which subsites the
        candidate source resolves; unresolved subsites are reported with missing
        statistics instead of being silently compared against a prior.
    :return: One record per subsite.
    :raises ValueError: If the two matrices do not have the same shape.
    """

    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} against {candidate.shape}.")

    from scipy.stats import spearmanr  # imported here to keep the module import light

    n_subsites = reference.shape[0]
    if observed_subsites is None:
        observed_subsites = [True] * n_subsites

    records: list[dict[str, float | str | bool]] = []
    for position in range(n_subsites):
        name = MEROPS_SUBSITES[position] if position < len(MEROPS_SUBSITES) else str(position)
        if not observed_subsites[position]:
            records.append(
                {
                    "subsite": name,
                    "observed": False,
                    "spearman": float("nan"),
                    "jensen_shannon_bits": float("nan"),
                    "top_residue_reference": "",
                    "top_residue_candidate": "",
                    "top_residue_agrees": False,
                }
            )
            continue

        reference_row = np.exp(reference[position])  # (A,)
        candidate_row = np.exp(candidate[position])  # (A,)
        correlation = spearmanr(reference_row, candidate_row).statistic
        top_reference = MEROPS_AMINO_ACIDS[int(np.argmax(reference_row))]
        top_candidate = MEROPS_AMINO_ACIDS[int(np.argmax(candidate_row))]
        records.append(
            {
                "subsite": name,
                "observed": True,
                "spearman": float(correlation) if np.isfinite(correlation) else float("nan"),
                "jensen_shannon_bits": _jensen_shannon_bits(reference_row, candidate_row),
                "top_residue_reference": top_reference,
                "top_residue_candidate": top_candidate,
                "top_residue_agrees": top_reference == top_candidate,
            }
        )
    return records


def window_from_sequence(
    sequence: str, bond: int, n_prime: int = 4, n_non_prime: int = 4, gap: str = "-"
) -> str:
    """Read the ``P4 … P4'`` window around a bond, padding with a gap character.

    :param sequence: Peptide sequence.
    :param bond: Index of the residue occupying ``P1``; the bond follows it.
    :param n_prime: Number of prime-side subsites.
    :param n_non_prime: Number of non-prime subsites.
    :param gap: Character marking a subsite outside the sequence.
    :return: A window string of length ``n_non_prime + n_prime``.
    """

    positions = range(bond - n_non_prime + 1, bond + n_prime + 1)
    return "".join(sequence[index] if 0 <= index < len(sequence) else gap for index in positions)
