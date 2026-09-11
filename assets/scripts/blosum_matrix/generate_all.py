#!/usr/bin/env python3
"""Regenerate all matrices under results/blosum_matrix."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

_SCRIPTS = Path(__file__).resolve().parent

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main() -> None:
    """Run export_blosum62.py then generate_ampblosum62.py."""

    for script in ("export_blosum62.py", "generate_ampblosum62.py"):
        path = _SCRIPTS / script
        typer.echo(f"==> {path.name}")
        subprocess.run([sys.executable, str(path)], check=True)


if __name__ == "__main__":
    app()
