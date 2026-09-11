"""Parse pre-aligned blocks of protein sequences."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO


@dataclass(frozen=True)
class Block:
    """A single pre-aligned, equal-width block of protein sequences."""

    path: Path
    ids: tuple[str, ...]
    sequences: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def width(self) -> int:
        return len(self.sequences[0])

    @property
    def size(self) -> int:
        return len(self.sequences)


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa", ".afa", ".aln"}:
        return "fasta"
    if suffix in {".sto", ".stk", ".stockholm"}:
        return "stockholm"
    raise ValueError(
        f"Unsupported block file extension {suffix!r} for {path}; "
        "expected aligned FASTA (.fa/.fasta/.faa/.afa/.aln) or Stockholm (.sto/.stk/.stockholm)"
    )


def parse_block(path: Path, min_block_width: int = 2) -> Block:
    """Parse one aligned FASTA / Stockholm file as a single block.

    Validates that all sequences share the same width and that the width is at
    least ``min_block_width``. Raises :class:`ValueError` for any violation.
    """

    path = Path(path)
    fmt = _detect_format(path)
    records = list(SeqIO.parse(str(path), fmt))
    if len(records) < 2:
        raise ValueError(f"block {path} has only {len(records)} sequence(s); need >= 2")

    ids = tuple(rec.id for rec in records)
    seqs = tuple(str(rec.seq).upper() for rec in records)

    widths = {len(s) for s in seqs}
    if len(widths) != 1:
        raise ValueError(
            f"block {path} has unequal sequence widths: {sorted(widths)} "
            "(unaligned input is rejected; one file must be one aligned block)"
        )

    (width,) = widths
    if width < min_block_width:
        raise ValueError(
            f"block {path} width {width} is below min_block_width {min_block_width}"
        )

    if len(set(ids)) != len(ids):
        raise ValueError(f"block {path} contains duplicate sequence ids")

    return Block(path=path, ids=ids, sequences=seqs)


def load_blocks(paths: Iterable[Path], min_block_width: int = 2) -> Iterator[Block]:
    """Iterate over block files, yielding one :class:`Block` per path."""

    for path in paths:
        yield parse_block(Path(path), min_block_width=min_block_width)
