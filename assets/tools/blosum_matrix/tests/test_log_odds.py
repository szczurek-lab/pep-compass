"""Tests for canonical Henikoff log-odds conversion.

The reference case is a uniform upper-triangular ``F_upper`` (every cell with
``a <= b`` equals 1). Closed-form expected values::

    T = 20 + C(20, 2) = 210
    p_a = (20 + 1) / (2 * 210) = 1/20 = 0.05  for every a
    q_aa = 1/210
    q_ab = 1/210  (for a != b)
    E_aa = p_a^2 = 1/400 = 0.0025
    E_ab = 2 * p_a * p_b = 1/200 = 0.005

Half-bits:
    s_aa = log2(q_aa / E_aa) / 0.5 = 2 * log2(400 / 210)
    s_ab = log2(q_ab / E_ab) / 0.5 = 2 * log2(200 / 210)

Bits:
    s_aa = log2(400 / 210)
    s_ab = log2(200 / 210)
"""

from __future__ import annotations

from math import log2

import numpy as np
import pytest

from blosum_matrix.alphabet import STANDARD_AA
from blosum_matrix.log_odds import counts_to_log_odds


def _uniform_upper() -> np.ndarray:
    F = np.zeros((20, 20))
    iu = np.triu_indices(20)
    F[iu] = 1.0
    return F


def test_log_odds_uniform_half_bits_no_round() -> None:
    F = _uniform_upper()
    scores = counts_to_log_odds(F, unit="half_bits", round_int=False)

    expected_diag = 2.0 * log2(400.0 / 210.0)
    expected_off = 2.0 * log2(200.0 / 210.0)

    # Symmetric by construction.
    assert np.allclose(scores, scores.T)
    diag = np.diag(scores)
    assert np.allclose(diag, expected_diag)
    off = scores - np.diag(diag)
    mask = ~np.eye(20, dtype=bool)
    assert np.allclose(off[mask], expected_off)


def test_log_odds_uniform_half_bits_integer_round() -> None:
    F = _uniform_upper()
    scores = counts_to_log_odds(F, unit="half_bits", round_int=True)

    expected_diag = round(2.0 * log2(400.0 / 210.0))
    expected_off = round(2.0 * log2(200.0 / 210.0))

    assert scores.dtype.kind == "i"
    assert np.all(np.diag(scores) == expected_diag)
    mask = ~np.eye(20, dtype=bool)
    assert np.all(scores[mask] == expected_off)


def test_log_odds_unit_bits_halves_half_bit_score() -> None:
    F = _uniform_upper()
    half = counts_to_log_odds(F, unit="half_bits", round_int=False)
    full = counts_to_log_odds(F, unit="bits", round_int=False)
    # half_bits = log2(...) / 0.5 = 2 * bits. So bits == half_bits / 2.
    assert np.allclose(full, half / 2.0)


def test_log_odds_offdiagonal_expectation_uses_two_p_a_p_b() -> None:
    """Sanity-check the ``2 * p_a * p_b`` factor for off-diagonal expectations.

    With a uniform ``F_upper`` the *unordered* off-diagonal probability is
    1/210. The score must be ``log2((1/210) / (2 * p_a * p_b))``. If the
    implementation forgot the factor of 2, the off-diagonal score would be
    ``log2((1/210) / (1/400))`` = ``log2(400/210)`` instead of the correct
    ``log2(200/210)``.
    """

    F = _uniform_upper()
    scores = counts_to_log_odds(F, unit="bits", round_int=False)

    correct_off = log2(200.0 / 210.0)
    wrong_off = log2(400.0 / 210.0)
    sample = scores[0, 1]
    assert sample == pytest.approx(correct_off)
    assert sample != pytest.approx(wrong_off)


def test_log_odds_zero_count_guard_reports_indices() -> None:
    F = _uniform_upper()
    F[3, 5] = 0.0  # 'D' x 'Q' on the canonical alphabet.

    with pytest.raises(ValueError) as exc:
        counts_to_log_odds(F, unit="half_bits")

    msg = str(exc.value)
    assert STANDARD_AA[3] in msg
    assert STANDARD_AA[5] in msg
    assert "(3, 5)" in msg
