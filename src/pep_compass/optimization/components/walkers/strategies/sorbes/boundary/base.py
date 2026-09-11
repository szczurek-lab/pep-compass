"""Boundary-stage contract for SORBES."""

from abc import ABC, abstractmethod


class BoundaryStrategy(ABC):
    """Resolve a proposed SORBES point against the supported domain."""

    @abstractmethod
    def __call__(self, previous, proposed, geometry):
        """Return accepted positions."""
