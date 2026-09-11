#!/usr/bin/env python3
"""Build AMP-BLOSUM62 from DBAASP length blocks into results/blosum_matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _paths import (  # noqa: E402
    DEFAULT_BLOCKS_DIR,
    DEFAULT_BLOCKS_GLOB,
    DEFAULT_OUT_DIR,
    REPO_ROOT,
    ensure_package_importable,
)

ensure_package_importable()

from blosum_matrix.blocks import load_blocks  # noqa: E402
from blosum_matrix.cluster_exact import cluster_block_exact  # noqa: E402
from blosum_matrix.counts import build_blosum  # noqa: E402
from blosum_matrix.log_odds import counts_to_log_odds  # noqa: E402
from blosum_matrix.ncbi import write_ncbi_matrix  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=False)

IDENTITY = 0.62


def _build_and_write(
    *,
    blocks_dir: Path,
    blocks_glob: str,
    out_path: Path,
    round_int: bool,
) -> None:
    paths = sorted(blocks_dir.glob(blocks_glob))
    if not paths:
        raise typer.BadParameter(
            f"no files matched {blocks_glob!r} under {blocks_dir}"
        )

    F_upper, stats = build_blosum(
        blocks=load_blocks(paths),
        identity=IDENTITY,
        cluster_fn=cluster_block_exact,
    )
    matrix = counts_to_log_odds(
        F_upper, unit="half_bits", round_int=round_int
    )
    try:
        blocks_label = blocks_dir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        blocks_label = blocks_dir
    comment = (
        "AMPBLOSUM62 computed by blosum-matrix "
        f"(canonical Henikoff; clustering=exact, identity={IDENTITY}, "
        f"unit=half_bits, round_int={round_int}); "
        f"blocks_dir={blocks_label}, glob={blocks_glob}; "
        f"blocks_seen={stats.blocks_seen}, blocks_kept={stats.blocks_kept}"
    )
    write_ncbi_matrix(out_path, matrix, comment=comment)
    typer.echo(
        " ".join(
            [
                f"out={out_path}",
                f"blocks_seen={stats.blocks_seen}",
                f"blocks_kept={stats.blocks_kept}",
                f"blocks_dropped_single_cluster={stats.blocks_dropped_single_cluster}",
                f"columns_used={stats.columns_used}",
                f"total_weight={stats.total_weight:.6f}",
                f"skipped_residues={stats.skipped_residues}",
            ]
        )
    )


@app.command()
def main(
    blocks_dir: Path = typer.Option(
        DEFAULT_BLOCKS_DIR,
        "--blocks-dir",
        help="Directory of equal-length AMP block FASTA files.",
    ),
    blocks_glob: str = typer.Option(
        DEFAULT_BLOCKS_GLOB,
        "--glob",
        help="Glob under --blocks-dir (one file = one block).",
    ),
    out_dir: Path = typer.Option(
        DEFAULT_OUT_DIR,
        "--out-dir",
        help="Directory for AMPBLOSUM62*.txt (created if missing).",
    ),
) -> None:
    """Build integer and float AMP-BLOSUM62 with exact clustering at T=0.62."""

    if not blocks_dir.is_dir():
        raise typer.BadParameter(f"--blocks-dir is not a directory: {blocks_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitkeep").touch(exist_ok=True)

    _build_and_write(
        blocks_dir=blocks_dir,
        blocks_glob=blocks_glob,
        out_path=out_dir / "AMPBLOSUM62.txt",
        round_int=True,
    )
    _build_and_write(
        blocks_dir=blocks_dir,
        blocks_glob=blocks_glob,
        out_path=out_dir / "AMPBLOSUM62_float.txt",
        round_int=False,
    )


if __name__ == "__main__":
    app()
