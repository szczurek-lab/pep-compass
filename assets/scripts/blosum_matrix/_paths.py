"""Shared paths for BLOSUM matrix data-generation scripts."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[2]
PACKAGE_ROOT = REPO_ROOT / "assets" / "tools" / "blosum_matrix"
PACKAGE_SRC = PACKAGE_ROOT / "src"

DEFAULT_OUT_DIR = REPO_ROOT / "results" / "blosum_matrix"
DEFAULT_BLOCKS_DIR = REPO_ROOT / "data" / "hydramp" / "dbaasp" / "blosum_blocks"
DEFAULT_BLOCKS_GLOB = "len_*.fasta"


def ensure_package_importable() -> None:
    """Put ``blosum_matrix`` on ``sys.path`` (src layout under assets/tools/blosum_matrix)."""

    path = str(PACKAGE_SRC)
    if path not in sys.path:
        sys.path.insert(0, path)
