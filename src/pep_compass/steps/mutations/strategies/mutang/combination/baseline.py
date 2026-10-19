"""Bounded Cartesian combination preserving baseline behaviour."""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING

from pep_compass.steps.mutations.strategies.mutang.base import (
    CombinationStrategy,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy as np

    from pep_compass.steps.mutations.strategies.mutang.schema import MutationOption


class BaselineCombination(CombinationStrategy):
    """Add parent residues, reduce oversized choices, and enumerate lazily."""

    def __init__(
        self, alphabet: Sequence[str], maximum_candidates: int | None = None
    ) -> None:
        """Initialize this object with the declared configuration.

        :param alphabet: Ordered residue alphabet.
        :type alphabet: Sequence[str]
        :param maximum_candidates: Maximum materialized candidates allowed by this local strategy.
        :type maximum_candidates: int | None
        :rtype: None
        """
        from pep_compass.utils.validation import validate_integer

        validate_integer(
            maximum_candidates,
            "BaselineCombination.maximum_candidates",
            1,
            optional=True,
        )
        self._configure(
            {"alphabet": alphabet, "maximum_candidates": maximum_candidates}
        )
        self.alphabet, self.maximum_candidates = alphabet, maximum_candidates

    def generate(
        self,
        sequence: str,
        options: tuple[MutationOption, ...],
        rng: np.random.Generator,
    ) -> Iterator[str]:
        """Generate.

        :param sequence: One peptide sequence.
        :type sequence: str
        :param options: Options.
        :type options: tuple[MutationOption, ...]
        :param rng: Run-owned random generator used for reproducible choices.
        :type rng: np.random.Generator
        :return: Result of this operation.
        :rtype: Iterator[str]
        """
        choices = {
            index: {self.alphabet.index(residue)}
            for index, residue in enumerate(sequence)
        }
        for option in options:
            choices[option.position].add(option.residue)
        while (
            self.maximum_candidates is not None
            and math.prod(map(len, choices.values())) > self.maximum_candidates
        ):
            position = max(choices, key=lambda key: len(choices[key]))
            parent = self.alphabet.index(sequence[position])
            alternatives = sorted(choices[position] - {parent})
            if not alternatives:
                break
            choices[position].remove(int(rng.choice(alternatives)))
        positions = sorted(choices)
        for residues in product(*(sorted(choices[position]) for position in positions)):
            yield "".join(self.alphabet[residue] for residue in residues)
