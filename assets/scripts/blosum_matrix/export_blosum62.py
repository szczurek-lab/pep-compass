#!/usr/bin/env python3
"""Export the official NCBI BLOSUM62 matrix (via BioPython) to results/blosum_matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import typer
from Bio.Align import substitution_matrices

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _paths import DEFAULT_OUT_DIR, ensure_package_importable  # noqa: E402

ensure_package_importable()

from blosum_matrix.alphabet import STANDARD_AA  # noqa: E402
from blosum_matrix.ncbi import write_ncbi_matrix  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    out_dir: Path = typer.Option(
        DEFAULT_OUT_DIR,
        "--out-dir",
        help="Directory for BLOSUM62.txt (created if missing).",
    ),
) -> None:
    """Write BioPython's bundled NCBI BLOSUM62 (20 standard amino acids)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitkeep").touch(exist_ok=True)

    loaded = substitution_matrices.load("BLOSUM62").select(STANDARD_AA)
    matrix = np.asarray(
        [[int(loaded[a, b]) for b in STANDARD_AA] for a in STANDARD_AA],
        dtype=np.int64,
    )

    biopython_version = getattr(
        __import__("Bio", fromlist=["__version__"]), "__version__", "unknown"
    )
    comment = (
        "Official NCBI BLOSUM62 via BioPython "
        f"Bio.Align.substitution_matrices.load('BLOSUM62'), "
        f"subset to {STANDARD_AA}; biopython={biopython_version}. "
        "Not computed from AMP blocks."
    )
    out_path = out_dir / "BLOSUM62.txt"
    write_ncbi_matrix(out_path, matrix, comment=comment)
    typer.echo(f"wrote {out_path}")


if __name__ == "__main__":
    app()
