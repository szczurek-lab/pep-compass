"""Integration test for the optional CD-HIT backend.

Skipped automatically unless both ``py-cdhit`` is importable and the ``cd-hit``
binary is on ``PATH``. On clusters where both are present, this test asserts
that the CD-HIT and exact backends partition a small contrived block the same
way.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

pycdhit = pytest.importorskip("pycdhit")

if shutil.which("cd-hit") is None:
    pytest.skip("cd-hit binary not on PATH", allow_module_level=True)

from blosum_matrix.blocks import Block  # noqa: E402
from blosum_matrix.cluster_cdhit import cluster_block_cdhit  # noqa: E402
from blosum_matrix.cluster_exact import cluster_block_exact  # noqa: E402


def _normalize(partition: dict[str, int]) -> tuple[frozenset[str], ...]:
    groups: dict[int, set[str]] = {}
    for seq_id, cid in partition.items():
        groups.setdefault(cid, set()).add(seq_id)
    return tuple(sorted((frozenset(g) for g in groups.values()), key=sorted))


def test_cdhit_matches_exact_on_small_block(tmp_path: Path) -> None:
    ids = ("s0", "s1", "s2")
    seqs = (
        "AAAAAAAAAAAAAAAAAAAA",
        "AAAAAAAAAAAAAAAAAAAA",
        "RRRRRRRRRRRRRRRRRRRR",
    )
    block = Block(path=tmp_path / "block.fasta", ids=ids, sequences=seqs)

    exact = cluster_block_exact(block, identity=0.7)
    cdhit = cluster_block_cdhit(block, identity=0.7, workdir=tmp_path)

    assert _normalize(exact) == _normalize(cdhit)
