"""Tests for the block parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from blosum_matrix.blocks import parse_block


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for seq_id, seq in records:
            handle.write(f">{seq_id}\n{seq}\n")
    return path


def test_parse_block_accepts_aligned_fasta(tmp_path: Path) -> None:
    path = _write_fasta(
        tmp_path / "block.fasta",
        [("s0", "AAAA"), ("s1", "AAAR"), ("s2", "RRRR")],
    )
    block = parse_block(path)
    assert block.size == 3
    assert block.width == 4
    assert block.ids == ("s0", "s1", "s2")


def test_parse_block_rejects_unequal_widths(tmp_path: Path) -> None:
    path = _write_fasta(
        tmp_path / "ragged.fasta",
        [("s0", "AAAA"), ("s1", "AAA")],
    )
    with pytest.raises(ValueError, match="unequal sequence widths"):
        parse_block(path)


def test_parse_block_rejects_single_sequence(tmp_path: Path) -> None:
    path = _write_fasta(tmp_path / "tiny.fasta", [("s0", "AAAA")])
    with pytest.raises(ValueError, match="need >= 2"):
        parse_block(path)


def test_parse_block_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_fasta(
        tmp_path / "dup.fasta",
        [("dup", "AAAA"), ("dup", "AAAR")],
    )
    with pytest.raises(ValueError, match="duplicate sequence ids"):
        parse_block(path)
