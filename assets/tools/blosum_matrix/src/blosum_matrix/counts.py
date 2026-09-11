"""Henikoff between-cluster weighted pair counting.

Counts are stored in an upper-triangular 20x20 ``F_upper`` matrix indexed by
``ARNDCQEGHILKMFPSTWYV``. ``F_upper[a, b]`` (``a <= b``) holds the unordered
pair count for ``(a, b)``. The lower triangle is mirrored only at output time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from blosum_matrix.alphabet import AA_INDEX, GAP_CHARS, STANDARD_AA
from blosum_matrix.blocks import Block

ClusterFn = Callable[..., dict[str, int]]


@dataclass
class BuildStats:
    blocks_seen: int = 0
    blocks_kept: int = 0
    blocks_dropped_single_cluster: int = 0
    columns_used: int = 0
    total_weight: float = 0.0
    skipped_residues: int = 0


def accumulate_block(
    block: Block,
    cluster_by_id: dict[str, int],
    F_upper: np.ndarray,
    stats: BuildStats | None = None,
) -> None:
    """Add Henikoff-weighted pair counts from ``block`` to ``F_upper``.

    For each column ``k`` and every unordered cluster pair ``(C_i, C_j)`` with
    ``i < j``, each cross-cluster sequence pair contributes
    ``1 / (|C_i| * |C_j|)`` to the unordered pair count of its residues. Pairs
    involving a gap or non-standard residue are skipped.
    """

    if F_upper.shape != (20, 20):
        raise ValueError(f"F_upper must be (20, 20); got {F_upper.shape}")

    indices_by_cluster: dict[int, list[int]] = defaultdict(list)
    for seq_idx, seq_id in enumerate(block.ids):
        if seq_id not in cluster_by_id:
            raise KeyError(f"sequence {seq_id!r} from block {block.path} is missing from cluster map")
        indices_by_cluster[cluster_by_id[seq_id]].append(seq_idx)

    cluster_ids_sorted = sorted(indices_by_cluster)
    cluster_sizes = {cid: len(indices_by_cluster[cid]) for cid in cluster_ids_sorted}

    if stats is not None:
        stats.columns_used += block.width

    seqs = block.sequences
    for k in range(block.width):
        for i_idx, ci in enumerate(cluster_ids_sorted):
            members_i = indices_by_cluster[ci]
            for cj in cluster_ids_sorted[i_idx + 1 :]:
                members_j = indices_by_cluster[cj]
                weight = 1.0 / (cluster_sizes[ci] * cluster_sizes[cj])
                for si in members_i:
                    a = seqs[si][k]
                    a_idx = AA_INDEX.get(a)
                    a_is_gap = a in GAP_CHARS
                    for sj in members_j:
                        b = seqs[sj][k]
                        b_is_gap = b in GAP_CHARS
                        if a_is_gap or b_is_gap:
                            if stats is not None:
                                stats.skipped_residues += 1
                            continue
                        b_idx = AA_INDEX.get(b)
                        if a_idx is None or b_idx is None:
                            if stats is not None:
                                stats.skipped_residues += 1
                            continue
                        lo, hi = (a_idx, b_idx) if a_idx <= b_idx else (b_idx, a_idx)
                        F_upper[lo, hi] += weight
                        if stats is not None:
                            stats.total_weight += weight


def build_blosum(
    blocks: Iterable[Block],
    identity: float,
    cluster_fn: ClusterFn,
    cluster_fn_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, BuildStats]:
    """Drive clustering and accumulation over a stream of blocks.

    ``cluster_fn`` receives ``(block, identity)`` and returns a mapping from
    sequence id to integer cluster id. The CD-HIT backend additionally needs a
    ``workdir``; pass it via ``cluster_fn_kwargs={"workdir": Path(...)}``.

    Returns ``(F_upper, stats)`` where ``F_upper`` is the upper-triangular
    20x20 count matrix (lower triangle remains zero).
    """

    F_upper = np.zeros((20, 20), dtype=np.float64)
    stats = BuildStats()
    extra = cluster_fn_kwargs or {}

    for block in blocks:
        stats.blocks_seen += 1
        cluster_by_id = cluster_fn(block, identity, **extra)  # type: ignore[arg-type]
        if len(set(cluster_by_id.values())) < 2:
            stats.blocks_dropped_single_cluster += 1
            continue
        accumulate_block(block, cluster_by_id, F_upper, stats=stats)
        stats.blocks_kept += 1

    assert STANDARD_AA == "ARNDCQEGHILKMFPSTWYV"
    return F_upper, stats
