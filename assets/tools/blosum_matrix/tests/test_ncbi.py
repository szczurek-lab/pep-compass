"""Tests for NCBI-format writer round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio.Align import substitution_matrices

from blosum_matrix.alphabet import STANDARD_AA
from blosum_matrix.log_odds import counts_to_log_odds
from blosum_matrix.ncbi import write_ncbi_matrix


def _uniform_upper() -> np.ndarray:
    F = np.zeros((20, 20))
    F[np.triu_indices(20)] = 1.0
    return F


def test_ncbi_roundtrip_loads_via_biopython(tmp_path: Path) -> None:
    scores = counts_to_log_odds(_uniform_upper(), unit="half_bits", round_int=True)
    out = tmp_path / "TEST.txt"
    write_ncbi_matrix(out, scores, comment="blosum-matrix unit test")

    loaded = substitution_matrices.read(str(out))
    for i, ai in enumerate(STANDARD_AA):
        for j, aj in enumerate(STANDARD_AA):
            assert int(loaded[ai, aj]) == int(scores[i, j])
