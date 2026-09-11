"""Optional CD-HIT backend.

Requires the ``cdhit`` extra (``py-cdhit``) and the ``cd-hit`` binary on
``PATH``. Both are GPL-2.0-only; see the package README. Failures are surfaced
loudly with no fallback.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from blosum_matrix.alphabet import GAP_CHARS
from blosum_matrix.blocks import Block


def _word_size_for(identity: float) -> int:
    """CD-HIT word size constraint as a function of identity threshold.

    CD-HIT refuses incompatible (c, n) combinations; this mirrors the upstream
    rules.
    """

    if identity >= 0.7:
        return 5
    if identity >= 0.6:
        return 4
    if identity >= 0.5:
        return 3
    if identity >= 0.4:
        return 2
    raise ValueError(
        f"identity={identity} is below CD-HIT's supported minimum (0.4); "
        "use --clustering exact for thresholds below 0.4"
    )


def _strip_gaps(seq: str) -> str:
    return "".join(c for c in seq if c not in GAP_CHARS)


def cluster_block_cdhit(block: Block, identity: float, workdir: Path) -> dict[str, int]:
    """Cluster a block's ungapped sequences with CD-HIT via py-cdhit.

    Writes the block's ungapped residues to a temp FASTA (IDs preserved, input
    order) inside ``workdir`` and runs ``cd-hit`` with deterministic options
    (``d=0, sc=1, T=1, n=_word_size_for(identity)``). Raises if ``cd-hit`` is
    not on ``PATH`` or if it returns nonzero / produces no ``.clstr`` output.
    """

    if shutil.which("cd-hit") is None:
        raise RuntimeError(
            "cd-hit binary not found on PATH; install CD-HIT (e.g. via your system "
            "package manager or conda) to use --clustering cdhit, or use --clustering exact"
        )

    try:
        from pycdhit import cd_hit, read_clstr
    except ImportError as exc:
        raise RuntimeError(
            "py-cdhit is not installed; add it with "
            "`uv sync --extra cdhit` "
            "or use --clustering exact"
        ) from exc

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    fasta_in = workdir / f"{block.name}.ungapped.fasta"
    fasta_out = workdir / f"{block.name}.cdhit"
    with fasta_in.open("w", encoding="utf-8") as handle:
        for seq_id, seq in zip(block.ids, block.sequences, strict=True):
            ungapped = _strip_gaps(seq)
            if not ungapped:
                raise ValueError(
                    f"sequence {seq_id!r} in block {block.path} is all gaps; cannot cluster"
                )
            handle.write(f">{seq_id}\n{ungapped}\n")

    try:
        cd_hit(
            i=str(fasta_in),
            o=str(fasta_out),
            c=identity,
            d=0,
            sc=1,
            T=1,
            n=_word_size_for(identity),
        )
    except Exception as exc:
        raise RuntimeError(
            f"cd-hit failed for block {block.path} at identity={identity}: {exc}"
        ) from exc

    clstr_path = fasta_out.with_suffix(fasta_out.suffix + ".clstr")
    if not clstr_path.exists():
        raise RuntimeError(
            f"cd-hit produced no .clstr file for block {block.path} (expected {clstr_path})"
        )

    df = read_clstr(str(clstr_path))
    if df.empty:
        raise RuntimeError(f"cd-hit produced empty .clstr for block {block.path}")

    seen: dict[int, int] = {}
    result: dict[str, int] = {}
    for seq_id, cluster_raw in zip(df["identifier"], df["cluster"], strict=True):
        cluster_int = int(cluster_raw)
        if cluster_int not in seen:
            seen[cluster_int] = len(seen)
        result[str(seq_id)] = seen[cluster_int]

    missing = [sid for sid in block.ids if sid not in result]
    if missing:
        raise RuntimeError(
            f"cd-hit output for block {block.path} is missing sequences: {missing[:5]}..."
        )
    return result
