"""Bounded Cartesian combination preserving baseline behaviour."""

from itertools import product
import math


class BaselineCombination:
    """Add parent residues, reduce oversized choices, and enumerate lazily."""

    def __init__(self, alphabet, maximum_candidates=None):
        self.alphabet, self.maximum_candidates = alphabet, maximum_candidates

    def generate(self, sequence, options, rng):
        choices = {index: {self.alphabet.index(residue)} for index, residue in enumerate(sequence)}
        for option in options:
            choices[option.position].add(option.residue)
        while self.maximum_candidates is not None and math.prod(map(len, choices.values())) > self.maximum_candidates:
            position = max(choices, key=lambda key: len(choices[key]))
            parent = self.alphabet.index(sequence[position])
            alternatives = sorted(choices[position] - {parent})
            if not alternatives:
                break
            choices[position].remove(int(rng.choice(alternatives)))
        positions = sorted(choices)
        for residues in product(*(sorted(choices[position]) for position in positions)):
            yield "".join(self.alphabet[residue] for residue in residues)
