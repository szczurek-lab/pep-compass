r"""Residue background distributions used as the null of specificity metrics.

A specificity metric answers "how different is this subsite from what would be
seen anyway". The reference distribution :math:`q` defines what "anyway" means.
Using a uniform :math:`q_a = 1/20` states that every residue is equally likely to
occupy a subsite, which no proteome satisfies: leucine is roughly an order of
magnitude more frequent than tryptophan. Under a uniform reference, a subsite
that merely reflects the composition of the substrate pool is scored as specific.

This module supplies the reference distributions and keeps their construction
separate from the metrics in
:mod:`~pep_compass.optimization.components.helpers.proteolysis.specificity`,
which accept :math:`q` as an argument. Acquisition of the source proteome is a
separate concern again and lives in
``assets/scripts/downloads/serum_proteolysis_calibration/download_human_proteome.py``.

Dimension symbols: ``A`` amino acids (20).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from pep_compass.optimization.components.helpers.proteolysis.cleavage import (
    MEROPS_AMINO_ACIDS,
)

# Location written by the acquisition script, relative to the repository root.
HUMAN_PROTEOME_COMPOSITION = Path("data") / "reference" / "human_proteome" / "composition.json"


def uniform_background(n_amino_acids: int = len(MEROPS_AMINO_ACIDS)) -> np.ndarray:
    """Return the uniform residue distribution.

    :param n_amino_acids: Size of the residue alphabet.
    :return: Uniform distribution over the alphabet, shape ``(A,)``.
    """

    return np.full(n_amino_acids, 1.0 / n_amino_acids)  # (A,)


def normalize_background(background: np.ndarray | None, n_amino_acids: int) -> np.ndarray:
    """Coerce a background argument into a normalized distribution.

    ``None`` resolves to the uniform distribution, so that every metric has one
    place where the default reference is decided.

    :param background: Residue distribution over the alphabet, or ``None``.
    :param n_amino_acids: Expected size of the residue alphabet.
    :return: Distribution summing to one, shape ``(A,)``.
    :raises ValueError: If the distribution has the wrong length, is not strictly
        positive, or does not sum to a positive value.
    """

    if background is None:
        return uniform_background(n_amino_acids)  # (A,)
    values = np.asarray(background, dtype=np.float64)  # (A,)
    if values.shape != (n_amino_acids,):
        raise ValueError(f"background must have shape ({n_amino_acids},), got {values.shape}.")
    if not np.all(values > 0.0):
        raise ValueError("background must be strictly positive; a zero makes the log-ratio undefined.")
    total = values.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("background must sum to a finite positive value.")
    return values / total  # (A,)


def amino_acid_background_from_fasta(
    path: str | Path,
    alphabet: str = MEROPS_AMINO_ACIDS,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute a residue background from a FASTA file.

    Residues outside ``alphabet`` are counted separately and excluded from the
    distribution, because the MEROPS matrices have no column for them. They are
    returned so that the exclusion is auditable rather than silent.

    :param path: FASTA file, optionally gzip-compressed.
    :param alphabet: Residue alphabet, in the column order of the MEROPS matrices.
    :return: Tuple of the residue distribution ``(A,)`` and the raw counts,
        including a ``"non_standard"`` entry and a ``"n_sequences"`` entry.
    :raises ValueError: If the file contains no residue of the alphabet.
    """

    path = Path(path)
    opener = _open_text(path)

    counts: Counter[str] = Counter()
    non_standard = 0
    n_sequences = 0
    with opener as handle:
        for line in handle:
            if line.startswith(">"):
                n_sequences += 1
                continue
            for residue in line.strip().upper():
                if residue in alphabet:
                    counts[residue] += 1
                else:
                    non_standard += 1

    values = np.array([counts.get(residue, 0) for residue in alphabet], dtype=np.float64)  # (A,)
    total = values.sum()
    if total <= 0.0:
        raise ValueError(f"{path} contains no residue of the alphabet {alphabet!r}.")

    raw_counts = {residue: int(counts.get(residue, 0)) for residue in alphabet}
    raw_counts["non_standard"] = int(non_standard)
    raw_counts["n_sequences"] = int(n_sequences)
    return values / total, raw_counts


def load_amino_acid_background(
    path: str | Path,
    alphabet: str = MEROPS_AMINO_ACIDS,
) -> np.ndarray:
    """Load a precomputed residue background written by the acquisition script.

    :param path: JSON file holding a ``frequencies`` object keyed by residue.
    :param alphabet: Residue alphabet, in the column order of the MEROPS matrices.
    :return: Residue distribution, shape ``(A,)``.
    :raises KeyError: If a residue of the alphabet is absent from the file.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frequencies = payload["frequencies"]
    values = np.array([float(frequencies[residue]) for residue in alphabet], dtype=np.float64)  # (A,)
    return normalize_background(values, len(alphabet))


def _open_text(path: Path):
    """Open a plain or gzip-compressed text file.

    :param path: File to open.
    :return: Context manager yielding text lines.
    """

    if path.suffix == ".gz":
        import gzip

        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")
