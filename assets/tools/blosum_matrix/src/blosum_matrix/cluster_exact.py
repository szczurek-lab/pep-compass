"""Canonical block-level clustering: greedy single-linkage on pairwise identity.

The block-level pairwise identity is computed directly from the aligned block as
``matching_non_gap_columns / non_gap_in_both_columns``. Cluster ids are assigned
in input order so that the partition is fully deterministic.
"""

from __future__ import annotations

from blosum_matrix.alphabet import GAP_CHARS
from blosum_matrix.blocks import Block


def _pairwise_identity(seq_a: str, seq_b: str) -> float:
    """Block-level identity on aligned sequences of equal length.

    Identity is ``matching_non_gap_columns / non_gap_in_both_columns``. Columns
    where either residue is a gap are excluded from both numerator and
    denominator. If no column has non-gap residues in both, identity is 0.0.
    """

    matches = 0
    both_non_gap = 0
    for a, b in zip(seq_a, seq_b, strict=True):
        if a in GAP_CHARS or b in GAP_CHARS:
            continue
        both_non_gap += 1
        if a == b:
            matches += 1
    if both_non_gap == 0:
        return 0.0
    return matches / both_non_gap


def cluster_block_exact(block: Block, identity: float) -> dict[str, int]:
    """Greedy single-linkage clustering on block-level identity.

    Two sequences are linked iff their block-level pairwise identity is
    ``>= identity``. Cluster ids are integers assigned in **input order**: the
    first sequence is in cluster 0, the first sequence not yet linked to it is
    in the next cluster, and so on.
    """

    n = block.size
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        # Keep the lower-indexed root so cluster ids follow input order.
        if ri < rj:
            parent[rj] = ri
        else:
            parent[ri] = rj

    seqs = block.sequences
    for i in range(n):
        for j in range(i + 1, n):
            if _pairwise_identity(seqs[i], seqs[j]) >= identity:
                union(i, j)

    cluster_id_by_root: dict[int, int] = {}
    result: dict[str, int] = {}
    for idx, seq_id in enumerate(block.ids):
        root = find(idx)
        if root not in cluster_id_by_root:
            cluster_id_by_root[root] = len(cluster_id_by_root)
        result[seq_id] = cluster_id_by_root[root]
    return result
