"""Strict reusable validation for Python and configuration declarations."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def validate_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    """Reject unknown declaration keys.

    :param value: Mapping to inspect.
    :param allowed: Permitted string keys.
    :param path: Diagnostic configuration path.
    :raises ValueError: If the mapping contains unknown keys.

    :type value: Mapping[str, Any]
    :type allowed: set[str]
    :type path: str
    :rtype: None
    """
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {path} keys: {sorted(unknown, key=str)}.")


def validate_integer(
    value: Any, path: str, minimum: int = 0, *, optional: bool = False
) -> None:
    """Require a bounded integer without coercing booleans or fractions.

    :param value: Value to validate.
    :param path: Diagnostic configuration path.
    :param minimum: Inclusive lower bound.
    :param optional: Whether null is accepted.
    :raises ValueError: If the value violates the integer contract.

    :type value: Any
    :type path: str
    :type minimum: int
    :type optional: bool
    :rtype: None
    """
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}.")


def validate_positive(value: Any, path: str, *, allow_zero: bool = False) -> None:
    """Require a finite positive real value.

    :param value: Value to validate.
    :param path: Diagnostic configuration path.
    :param allow_zero: Accept zero as an inclusive lower boundary.
    :raises ValueError: If the numeric constraint is violated.

    :type value: Any
    :type path: str
    :type allow_zero: bool
    :rtype: None
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        raise ValueError(
            f"{path} must be finite and {'non-negative' if allow_zero else 'positive'}."
        )


def validate_sequences(sequences: Sequence[str]) -> tuple[object, ...]:
    """Return non-empty peptide strings before preparing any model resources.

    :param sequences: Iterable of non-empty peptide strings.
    :return: Independent tuple of validated input sequences.
    :raises ValueError: If the input is absent, empty or contains non-string values.

    :type sequences: Sequence[str]
    :rtype: tuple[object, ...]
    """
    if sequences is None or isinstance(sequences, (str, bytes)):
        raise ValueError("Provide a non-empty collection of peptide strings.")
    values = tuple(sequences)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("Provide a non-empty collection of peptide strings.")
    return values
