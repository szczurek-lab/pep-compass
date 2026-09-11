"""BLOSUM helpers used by optional optimizer diversity filtering."""

import blosum

_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_CACHE: dict[int, blosum.BLOSUM] = {}


def load_blosum(number: int) -> blosum.BLOSUM:
    """Load and cache a BLOSUM matrix."""
    if number not in _CACHE:
        _CACHE[number] = blosum.BLOSUM(number)
    return _CACHE[number]


def blosum_score(
    first_sequence: str,
    second_sequence: str,
    matrix: blosum.BLOSUM,
) -> float:
    """Return the sum of per-position BLOSUM substitution scores.

    Characters outside the standard amino-acid alphabet, including padding,
    contribute zero. For sequences of different lengths, only their shared
    prefix is scored.
    """
    return float(
        sum(
            matrix[first][second]
            for first, second in zip(first_sequence, second_sequence)
            if first in _AMINO_ACIDS and second in _AMINO_ACIDS
        )
    )
