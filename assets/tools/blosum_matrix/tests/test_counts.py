"""Tests for canonical Henikoff between-cluster weighted pair counting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blosum_matrix.alphabet import AA_INDEX
from blosum_matrix.blocks import Block
from blosum_matrix.counts import BuildStats, accumulate_block

A = AA_INDEX["A"]
R = AA_INDEX["R"]


def _block(seqs: list[tuple[str, str]]) -> Block:
    ids, sequences = zip(*seqs, strict=True)
    return Block(path=Path("/tmp/fake.fasta"), ids=tuple(ids), sequences=tuple(sequences))


def test_henikoff_toy_2aa_block_hand_derived_counts() -> None:
    """Hand-derived F_upper for a 3-sequence, width-3 block restricted to {A, R}.

    Sequences::

        s0 = A A R   (cluster 0)
        s1 = A R A   (cluster 0)
        s2 = R R R   (cluster 1)

    Cluster sizes are 2 and 1, so each cross-cluster sequence pair contributes
    weight ``1 / (2 * 1) = 0.5``.

    * Column 0: cross-cluster pairs (s0, s2)=(A, R) and (s1, s2)=(A, R) → 1.0 to F[A, R].
    * Column 1: (s0, s2)=(A, R) → 0.5 to F[A, R]; (s1, s2)=(R, R) → 0.5 to F[R, R].
    * Column 2: (s0, s2)=(R, R) → 0.5 to F[R, R]; (s1, s2)=(A, R) → 0.5 to F[A, R].

    Expected upper triangle on the {A, R} submatrix: F[A, A]=0, F[A, R]=2.0,
    F[R, R]=1.0.
    """

    block = _block(
        [
            ("s0", "AAR"),
            ("s1", "ARA"),
            ("s2", "RRR"),
        ]
    )
    cluster_by_id = {"s0": 0, "s1": 0, "s2": 1}

    F = np.zeros((20, 20))
    stats = BuildStats()
    accumulate_block(block, cluster_by_id, F, stats=stats)

    assert F[A, A] == 0.0
    assert F[A, R] == pytest.approx(2.0)
    assert F[R, R] == pytest.approx(1.0)

    # All cells outside the {A, R} submatrix must be untouched.
    mask = np.ones((20, 20), dtype=bool)
    for i, j in [(A, A), (A, R), (R, R)]:
        mask[i, j] = False
    assert np.all(F[mask] == 0.0)

    # Only the upper triangle is mutated: F[R, A] must remain zero (the call
    # always writes to F[min, max]).
    assert F[R, A] == 0.0

    # Stats: 3 columns visited; total weight equals F.sum() (upper-only) since
    # we add weight for every counted residue pair into one upper cell.
    assert stats.columns_used == 3
    assert stats.total_weight == pytest.approx(F.sum())
    assert stats.skipped_residues == 0


def test_same_cluster_pairs_are_not_counted() -> None:
    block = _block(
        [
            ("s0", "AA"),
            ("s1", "AA"),
        ]
    )
    cluster_by_id = {"s0": 0, "s1": 0}

    F = np.zeros((20, 20))
    accumulate_block(block, cluster_by_id, F)

    assert F.sum() == 0.0


def test_non_standard_and_gap_residues_are_skipped() -> None:
    block = _block(
        [
            ("s0", "X-A"),
            ("s1", "AAA"),
        ]
    )
    cluster_by_id = {"s0": 0, "s1": 1}

    F = np.zeros((20, 20))
    stats = BuildStats()
    accumulate_block(block, cluster_by_id, F, stats=stats)

    # Only the final column has standard residues for both sequences: (A, A).
    assert F[A, A] == pytest.approx(1.0)
    assert F.sum() == pytest.approx(1.0)
    # Two columns produced a skip: 'X' is non-standard, '-' is a gap.
    assert stats.skipped_residues == 2
