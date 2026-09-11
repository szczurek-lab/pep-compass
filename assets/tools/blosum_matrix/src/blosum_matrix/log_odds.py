"""Convert Henikoff unordered pair counts into a half-bit log-odds matrix."""

from __future__ import annotations

from typing import Literal

import numpy as np

from blosum_matrix.alphabet import STANDARD_AA

Unit = Literal["half_bits", "bits"]


def _unit_scale(unit: Unit) -> float:
    if unit == "half_bits":
        return 0.5
    if unit == "bits":
        return 1.0
    raise ValueError(f"unit must be 'half_bits' or 'bits'; got {unit!r}")


def counts_to_log_odds(
    F_upper: np.ndarray,
    unit: Unit = "half_bits",
    round_int: bool = True,
) -> np.ndarray:
    """Canonical Henikoff log-odds with correct unordered marginals.

    With upper-triangular ``F_upper`` (``F_upper[a, b]`` for ``a <= b``):

    * ``T = F_upper.sum()`` (total unordered pair mass)
    * ``q[a, b] = F_upper[a, b] / T`` for ``a <= b``; ``q[b, a] := q[a, b]``
    * ``p[a] = q[a, a] + 0.5 * sum_{b != a} q[a, b]`` (canonical marginal)
    * Expected pair frequency under independence:
      ``E[a, a] = p[a] ** 2``; ``E[a, b] = 2 * p[a] * p[b]`` for ``a != b``.
    * Score: ``s[a, b] = log2(q[a, b] / E[a, b]) / unit_scale``, where
      ``unit_scale = 0.5`` for half-bits (NCBI/BLAST), ``1.0`` for bits.
    * Round to nearest integer when ``round_int`` is True.

    Raises if any cell of ``q`` is zero (canonical Henikoff has no pseudocount).
    """

    if F_upper.shape != (20, 20):
        raise ValueError(f"F_upper must be (20, 20); got {F_upper.shape}")
    if np.any(F_upper < 0):
        raise ValueError("F_upper contains negative entries")

    total = float(F_upper.sum())
    if total <= 0.0:
        raise ValueError(
            "F_upper has zero total mass; no between-cluster pairs were counted"
        )

    upper_mask = np.triu(np.ones((20, 20), dtype=bool))
    zero_cells = np.argwhere((F_upper == 0.0) & upper_mask)
    if len(zero_cells) > 0:
        i, j = int(zero_cells[0, 0]), int(zero_cells[0, 1])
        a, b = STANDARD_AA[i], STANDARD_AA[j]
        n_zero = len(zero_cells)
        raise ValueError(
            f"zero-count cell at ({a!r}, {b!r}) (index ({i}, {j})); "
            f"{n_zero} upper-triangle cells have zero mass. "
            "Canonical Henikoff has no pseudocount; supply more blocks or longer blocks."
        )

    F_sym = F_upper + F_upper.T - np.diag(np.diag(F_upper))
    q_sym = F_sym / total

    p = (F_sym.sum(axis=1) + np.diag(F_sym)) / (2.0 * total)

    expected = 2.0 * np.outer(p, p)
    np.fill_diagonal(expected, p * p)

    u = _unit_scale(unit)
    scores = np.log2(q_sym / expected) / u

    if round_int:
        scores = np.rint(scores).astype(np.int64)
    return scores
