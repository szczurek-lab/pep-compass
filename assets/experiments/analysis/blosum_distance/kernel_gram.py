"""Mutation-kernel Gram matrix G_K and its symmetric square root.

Implements the ambient metric of PepCompass appendix D.3 for the special case of
a *residue-level* mutation kernel with no positional coupling:

    S      symmetric 20x20 substitution-score matrix (AMP-BLOSUM or BLOSUM62)
    c      = -min(eig(S))                      so that S + cI is PSD
    K      = exp(beta * (S + cI))              ENTRYWISE, not expm
    G_res  = K                                 20x20, strictly PD (Schur product thm)
    G_tok  = diag_block(1, G_res)               21x21, identity on the padding token
    G_K    = I_L (x) G_tok                      (L*21) x (L*21) ambient Gram matrix

``G_K`` is block diagonal over sequence positions: the same 21x21 residue block is
applied independently at each of the ``L`` positions. This is a *choice* permitted
by D.3 (which allows, but does not require, positional coupling) and it matches the
block structure ``(I_L (x) sqrt(M)) J`` used by the repo's own ambient-metric code.

The ambient coordinate order is position-major, ``index = position * 21 + token``,
matching ``rearrange(x, "b seq vocab -> b (seq vocab)")`` in the HydrAMP adapter and
``left_vectors.reshape(max_len, alphabet_size, -1)`` in MUTANG's scoring stage.

Token 0 is the HydrAMP padding symbol (``PADDED_ALPHABET = " ACDEFGHIKLMNPQRSTVWY"``).
It is left Euclidean: MUTANG never proposes it (``scores[:, 1:]``), and giving it a
kernel weight would change the norm of padding directions for no biological reason.
"""

from __future__ import annotations

import numpy as np

#: PepCompass residue order for the 20x20 score matrices.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: HydrAMP ambient token alphabet; index 0 is the padding symbol.
PADDED_ALPHABET = " " + AMINO_ACIDS


def read_score_matrix(path: str, alphabet: str = AMINO_ACIDS) -> np.ndarray:
    """Read an NCBI-format substitution matrix and reorder it to ``alphabet``.

    Drops ``#`` comment lines; the first remaining line is the column header and
    each later line is ``RESIDUE v1 v2 ...``.

    :param path: Path to the matrix file.
    :param alphabet: Residue order of the returned matrix.
    :return: Symmetric ``(20, 20)`` score matrix in ``alphabet`` order.
    """
    rows: dict[str, dict[str, float]] = {}
    columns: list[str] | None = None
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if columns is None:
                columns = fields
                continue
            residue, values = fields[0], [float(value) for value in fields[1:]]
            rows[residue] = dict(zip(columns, values))

    scores = np.array(
        [[rows[a][b] for b in alphabet] for a in alphabet],
        dtype=np.float64,
    )
    return 0.5 * (scores + scores.T)


def diagonal_shift(scores: np.ndarray) -> float:
    """Return ``c = -min(eig(S))``, the shift that makes ``S + cI`` PSD.

    :param scores: Symmetric score matrix.
    :return: Non-negative shift ``c``.
    """
    return float(max(0.0, -np.linalg.eigvalsh(scores).min()))


def mutation_kernel(scores: np.ndarray, *, beta: float = 4.0) -> np.ndarray:
    """Build ``K = exp(beta * (S + cI))`` entrywise.

    The shift ``c`` makes ``S + cI`` PSD, so the entrywise exponential is PD by the
    Schur product theorem (a convergent sum of Hadamard powers of a PSD matrix).
    The literal matrix exponential ``expm(beta * S)`` is *not* usable in its place:
    it carries negative off-diagonal entries, which would encode negative similarity
    between chemically distinct residues.

    :param scores: Symmetric ``(A, A)`` score matrix.
    :param beta: Inverse-temperature of the kernel.
    :return: Symmetric PD ``(A, A)`` kernel matrix.
    """
    shifted = scores + diagonal_shift(scores) * np.eye(scores.shape[0])
    kernel = np.exp(beta * shifted)
    return 0.5 * (kernel + kernel.T)


def symmetric_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric PSD square root by eigendecomposition.

    :param matrix: Symmetric PD matrix.
    :return: Symmetric matrix ``B`` with ``B @ B == matrix``.
    :raises ValueError: If ``matrix`` is not strictly positive definite.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if eigenvalues.min() <= 0.0:
        raise ValueError(
            f"matrix is not strictly positive definite: min eigenvalue "
            f"{eigenvalues.min():.6g}"
        )
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def pad_residue_block(residue_block: np.ndarray) -> np.ndarray:
    """Embed a ``(20, 20)`` residue block into a ``(21, 21)`` token block.

    The padding token keeps a unit diagonal entry and no coupling to residues, so
    the padding coordinate stays Euclidean.

    :param residue_block: ``(20, 20)`` residue-level matrix.
    :return: ``(21, 21)`` token-level matrix.
    """
    width = len(PADDED_ALPHABET)
    block = np.eye(width, dtype=np.float64)
    block[1:, 1:] = residue_block
    return block


def ambient_gram(token_block: np.ndarray, max_len: int = 25) -> np.ndarray:
    """Lift a token block to the full ambient Gram matrix ``I_L (x) block``.

    :param token_block: ``(21, 21)`` token-level matrix.
    :param max_len: Ambient sequence length ``L``.
    :return: ``(L * 21, L * 21)`` block-diagonal matrix.
    """
    return np.kron(np.eye(max_len, dtype=np.float64), token_block)


def build_gram(
    path: str,
    *,
    beta: float = 4.0,
    max_len: int = 25,
) -> dict[str, np.ndarray | float]:
    """Build the full G_K / G_K^(1/2) stack from a score-matrix file.

    :param path: Path to an NCBI-format substitution matrix.
    :param beta: Kernel inverse-temperature.
    :param max_len: Ambient sequence length ``L``.
    :return: Dict with ``scores``, ``shift``, ``kernel``, ``kernel_sqrt``,
        ``token_block``, ``token_block_sqrt``, ``min_eigenvalue``,
        ``condition_number`` and ``sqrt_residual``.
    """
    scores = read_score_matrix(path)
    shift = diagonal_shift(scores)
    kernel = mutation_kernel(scores, beta=beta)
    kernel_sqrt = symmetric_sqrt(kernel)
    eigenvalues = np.linalg.eigvalsh(kernel)
    return {
        "scores": scores,
        "shift": shift,
        "beta": beta,
        "max_len": max_len,
        "kernel": kernel,
        "kernel_sqrt": kernel_sqrt,
        "token_block": pad_residue_block(kernel),
        "token_block_sqrt": pad_residue_block(kernel_sqrt),
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "sqrt_residual": float(np.abs(kernel_sqrt @ kernel_sqrt - kernel).max()),
    }


def validate(bundle: dict, *, tolerance: float = 1e-8) -> dict[str, float]:
    """Assert the step-2 correctness properties of a ``build_gram`` bundle.

    Checks strict positive definiteness, symmetry of the square root, that the
    square root reproduces the kernel, that the padding coordinate is untouched,
    and that ``K = I`` returns the identity map exactly.

    :param bundle: Output of :func:`build_gram`.
    :param tolerance: Relative tolerance for the square-root residual.
    :return: Dict of the measured residuals.
    """
    kernel = bundle["kernel"]
    root = bundle["kernel_sqrt"]

    assert np.allclose(kernel, kernel.T, atol=0.0, rtol=0.0), "kernel not symmetric"
    assert bundle["min_eigenvalue"] > 0.0, (
        f"kernel not strictly PD: min eigenvalue {bundle['min_eigenvalue']:.6g}"
    )

    symmetry = float(np.abs(root - root.T).max() / np.abs(root).max())
    assert symmetry <= tolerance, f"square root not symmetric: {symmetry:.6g}"

    scale = float(np.abs(kernel).max())
    relative = bundle["sqrt_residual"] / scale
    assert relative <= tolerance, (
        f"sqrt(K) @ sqrt(K) != K: relative residual {relative:.6g}"
    )

    # Padding coordinate must stay Euclidean and uncoupled.
    token = bundle["token_block_sqrt"]
    assert token[0, 0] == 1.0, "padding diagonal changed"
    assert np.all(token[0, 1:] == 0.0) and np.all(token[1:, 0] == 0.0), (
        "padding token coupled to residues"
    )

    # K = I must return the identity map exactly (bit-identical Euclidean path).
    identity = np.eye(kernel.shape[0])
    identity_root = symmetric_sqrt(identity)
    assert np.array_equal(identity_root, identity), (
        "sqrt(I) is not exactly the identity"
    )
    identity_ambient = ambient_gram(pad_residue_block(identity_root), max_len=3)
    assert np.array_equal(identity_ambient, np.eye(3 * len(PADDED_ALPHABET))), (
        "I_L (x) sqrt(I) is not exactly the ambient identity"
    )

    return {
        "min_eigenvalue": bundle["min_eigenvalue"],
        "max_eigenvalue": bundle["max_eigenvalue"],
        "condition_number": bundle["condition_number"],
        "sqrt_residual_absolute": bundle["sqrt_residual"],
        "sqrt_residual_relative": relative,
        "sqrt_symmetry_residual": symmetry,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    matrix_path = sys.argv[1] if len(sys.argv) > 1 else str(
        next(
            parent / "src/pep_compass/optimization/components/helpers/substitution_matrices/matrices/AMPBLOSUM62_float.txt"
            for parent in Path(__file__).resolve().parents
            if (parent / "src/pep_compass/optimization/components/helpers/substitution_matrices/matrices/AMPBLOSUM62_float.txt").is_file()
        )
    )
    result = build_gram(matrix_path, beta=4.0)
    for key, value in validate(result).items():
        print(f"{key}\t{value:.6g}")
