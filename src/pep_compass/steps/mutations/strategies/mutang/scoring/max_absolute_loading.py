"""Baseline MUTANG score matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pep_compass.steps.mutations.strategies.mutang.base import (
    ScoringStrategy,
)

if TYPE_CHECKING:
    import torch

    from pep_compass.steps.mutations.strategies.mutang.schema import SelectedDirections


class MaxAbsoluteLoading(ScoringStrategy):
    """Return ``max_j abs(U[(l,a),j])`` for selected directions."""

    def __init__(self, max_len: int = 25, alphabet_size: int = 21) -> None:
        """Initialize this object with the declared configuration.

        :param max_len: Max len.
        :type max_len: int
        :param alphabet_size: Alphabet size.
        :type alphabet_size: int
        :rtype: None
        """
        self._configure({"max_len": max_len, "alphabet_size": alphabet_size})
        self.max_len, self.alphabet_size = max_len, alphabet_size

    def score(
        self, left_vectors: torch.Tensor, selected: SelectedDirections
    ) -> torch.Tensor:
        """Score.

        :param left_vectors: Left vectors.
        :type left_vectors: torch.Tensor
        :param selected: Selected.
        :type selected: SelectedDirections
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        table = left_vectors[:, selected.indices].abs()
        return table.reshape(self.max_len, self.alphabet_size, -1).amax(dim=-1)
