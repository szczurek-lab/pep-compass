"""Standard 20 amino-acid alphabet used throughout the package."""

from __future__ import annotations

STANDARD_AA: str = "ARNDCQEGHILKMFPSTWYV"
"""Canonical NCBI-style row/column order for the BLOSUM output."""

GAP_CHARS: frozenset[str] = frozenset("-.")
"""Characters interpreted as gaps in aligned blocks."""

AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(STANDARD_AA)}
"""Map from amino-acid letter to row/column index."""
