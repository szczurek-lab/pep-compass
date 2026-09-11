"""Command-line entry point for ``blosum-matrix``."""

from __future__ import annotations

import tempfile
from enum import Enum
from pathlib import Path

import typer

from blosum_matrix.blocks import load_blocks
from blosum_matrix.cluster_exact import cluster_block_exact
from blosum_matrix.counts import build_blosum
from blosum_matrix.log_odds import counts_to_log_odds
from blosum_matrix.ncbi import write_ncbi_matrix


class Clustering(str, Enum):
    exact = "exact"
    cdhit = "cdhit"


class Unit(str, Enum):
    half_bits = "half_bits"
    bits = "bits"


app = typer.Typer(
    name="blosum-matrix",
    help=(
        "Build a BLOSUM substitution matrix from pre-aligned ungapped blocks "
        "using the canonical Henikoff & Henikoff (1992) procedure. "
        "One input file is treated as one block."
    ),
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _main() -> None:
    """Force Typer to keep the ``build`` subcommand even when it's the only one."""


def _collect_block_paths(
    blocks: list[Path] | None,
    blocks_dir: Path | None,
    glob: str,
) -> list[Path]:
    blocks = blocks or []
    if blocks and blocks_dir is not None:
        raise typer.BadParameter(
            "--blocks and --blocks-dir are mutually exclusive; pass one or the other"
        )
    if blocks:
        for path in blocks:
            if not path.is_file():
                raise typer.BadParameter(f"block file does not exist: {path}")
        return list(blocks)
    if blocks_dir is None:
        raise typer.BadParameter(
            "supply --blocks PATH (repeatable) or --blocks-dir PATH with --glob"
        )
    if not blocks_dir.is_dir():
        raise typer.BadParameter(f"--blocks-dir is not a directory: {blocks_dir}")
    found = sorted(blocks_dir.glob(glob))
    if not found:
        raise typer.BadParameter(
            f"no files matched {glob!r} under {blocks_dir}"
        )
    return found


@app.command("build")
def build(
    blocks: list[Path] = typer.Option(
        None,
        "--blocks",
        help="Path to one aligned block file. Repeat to pass multiple blocks.",
    ),
    blocks_dir: Path | None = typer.Option(
        None,
        "--blocks-dir",
        help="Directory containing one block per file. Use with --glob.",
    ),
    glob: str = typer.Option(
        "*.fasta",
        "--glob",
        help="Glob pattern applied under --blocks-dir.",
    ),
    out_matrix: Path = typer.Option(
        ...,
        "--out-matrix",
        help="Output path for the NCBI-format substitution matrix.",
    ),
    identity: float = typer.Option(
        0.62,
        "--identity",
        min=0.0,
        max=1.0,
        help="Clustering identity threshold T (e.g. 0.62 for BLOSUM62).",
    ),
    clustering: Clustering = typer.Option(
        Clustering.exact,
        "--clustering",
        help="Clustering backend. 'exact' is canonical and has no GPL deps.",
    ),
    unit: Unit = typer.Option(
        Unit.half_bits,
        "--unit",
        help="Score unit. 'half_bits' is NCBI/BLAST convention.",
    ),
    no_round: bool = typer.Option(
        False,
        "--no-round",
        help="Keep float scores instead of rounding to the NCBI-style integers.",
    ),
    min_block_width: int = typer.Option(
        2,
        "--min-block-width",
        min=1,
        help="Reject blocks whose alignment width is below this value.",
    ),
    workdir: Path | None = typer.Option(
        None,
        "--workdir",
        help="Directory for CD-HIT artifacts (only used with --clustering cdhit).",
    ),
) -> None:
    """Build a BLOSUM matrix and write it to ``--out-matrix``."""

    block_paths = _collect_block_paths(blocks, blocks_dir, glob)
    block_iter = load_blocks(block_paths, min_block_width=min_block_width)

    label = f"BLOSUM{int(round(identity * 100))}"

    if clustering is Clustering.exact:
        cluster_fn = lambda block, t: cluster_block_exact(block, t)  # noqa: E731
        cluster_kwargs: dict[str, object] = {}
        cleanup_tmp: tempfile.TemporaryDirectory | None = None
    else:
        from blosum_matrix.cluster_cdhit import cluster_block_cdhit

        if workdir is None:
            cleanup_tmp = tempfile.TemporaryDirectory(prefix="blosum-matrix-")
            workdir_path = Path(cleanup_tmp.name)
        else:
            cleanup_tmp = None
            workdir_path = workdir
            workdir_path.mkdir(parents=True, exist_ok=True)

        def _cdhit(block, t, *, workdir: Path):  # type: ignore[no-redef]
            return cluster_block_cdhit(block, t, workdir=workdir)

        cluster_fn = _cdhit
        cluster_kwargs = {"workdir": workdir_path}

    try:
        F_upper, stats = build_blosum(
            blocks=block_iter,
            identity=identity,
            cluster_fn=cluster_fn,
            cluster_fn_kwargs=cluster_kwargs,
        )
    finally:
        if "cleanup_tmp" in locals() and cleanup_tmp is not None:
            cleanup_tmp.cleanup()

    matrix = counts_to_log_odds(F_upper, unit=unit.value, round_int=not no_round)

    comment = (
        f"{label} computed by blosum-matrix "
        f"(canonical Henikoff; clustering={clustering.value}, identity={identity}, "
        f"unit={unit.value}); "
        f"blocks_seen={stats.blocks_seen}, blocks_kept={stats.blocks_kept}"
    )
    write_ncbi_matrix(out_matrix, matrix, comment=comment)

    typer.echo(
        " ".join(
            [
                f"label={label}",
                f"backend={clustering.value}",
                f"blocks_seen={stats.blocks_seen}",
                f"blocks_kept={stats.blocks_kept}",
                f"blocks_dropped_single_cluster={stats.blocks_dropped_single_cluster}",
                f"columns_used={stats.columns_used}",
                f"total_weight={stats.total_weight:.6f}",
                f"skipped_residues={stats.skipped_residues}",
                f"threshold={identity}",
                f"unit={unit.value}",
                f"out={out_matrix}",
            ]
        )
    )
