"""Write substitution matrices in NCBI/BLAST text format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blosum_matrix.alphabet import STANDARD_AA


def write_ncbi_matrix(
    path: Path,
    matrix: np.ndarray,
    alphabet: str = STANDARD_AA,
    comment: str | None = None,
) -> None:
    """Emit ``matrix`` in NCBI/BLAST text format on ``alphabet``.

    The resulting file is loadable via :func:`Bio.Align.substitution_matrices.read`.
    Integer matrices are written as integers; floats are written with 6
    decimals.
    """

    matrix = np.asarray(matrix)
    n = len(alphabet)
    if matrix.shape != (n, n):
        raise ValueError(f"matrix shape {matrix.shape} does not match alphabet length {n}")

    is_int = np.issubdtype(matrix.dtype, np.integer)
    if is_int:
        cells = [[f"{int(v):4d}" for v in row] for row in matrix]
        col_width = 4
    else:
        cells = [[f"{float(v):10.6f}" for v in row] for row in matrix]
        col_width = 10

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if comment:
            for line in comment.splitlines():
                handle.write(f"# {line}\n")
        header_cells = "".join(f"{ch:>{col_width}}" for ch in alphabet)
        handle.write(f"  {header_cells}\n")
        for i, row_char in enumerate(alphabet):
            handle.write(f"{row_char} {''.join(cells[i])}\n")
