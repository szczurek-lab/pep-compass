"""Base contract for filters without reusable scalar ranking."""

from abc import ABC

from pep_compass.optimization.components.filters.base import Filter


class DirectFilter(Filter, ABC):
    """Directly restrict or structurally transform a candidate batch."""
