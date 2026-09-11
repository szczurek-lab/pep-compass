"""Tests for the canonical (exact) block-level clustering backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from blosum_matrix.blocks import Block
from blosum_matrix.cluster_exact import _pairwise_identity, cluster_block_exact


def _block(seqs: list[tuple[str, str]]) -> Block:
    ids, sequences = zip(*seqs, strict=True)
    return Block(path=Path("/tmp/fake.fasta"), ids=tuple(ids), sequences=tuple(sequences))


def test_pairwise_identity_ignores_gap_columns() -> None:
    assert _pairwise_identity("AAAA", "AAAA") == 1.0
    assert _pairwise_identity("AAAA", "AAAR") == pytest.approx(0.75)
    assert _pairwise_identity("AAAA", "RRRR") == 0.0
    # Gaps reduce the denominator; remaining columns are perfectly matched.
    assert _pairwise_identity("A-AA", "AAAA") == 1.0
    # No overlapping non-gap columns -> identity is 0.
    assert _pairwise_identity("--AA", "AA--") == 0.0


def test_cluster_block_exact_partitions_at_threshold() -> None:
    block = _block(
        [
            ("s0", "AAAA"),
            ("s1", "AAAA"),
            ("s2", "AAAR"),
            ("s3", "RRRR"),
        ]
    )

    at_05 = cluster_block_exact(block, identity=0.5)
    assert at_05 == {"s0": 0, "s1": 0, "s2": 0, "s3": 1}

    at_08 = cluster_block_exact(block, identity=0.8)
    assert at_08 == {"s0": 0, "s1": 0, "s2": 1, "s3": 2}

    at_10 = cluster_block_exact(block, identity=1.0)
    assert at_10 == {"s0": 0, "s1": 0, "s2": 1, "s3": 2}


def test_cluster_ids_follow_input_order() -> None:
    # The first sequence not linked to s0 must become cluster 1, then 2, etc.
    block = _block(
        [
            ("alpha", "RRRR"),
            ("beta", "RRRR"),
            ("gamma", "AAAA"),
        ]
    )
    result = cluster_block_exact(block, identity=1.0)
    assert result == {"alpha": 0, "beta": 0, "gamma": 1}
