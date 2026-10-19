"""MOVE candidate scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.steps.filters.strategies.ranked.scoring.helpers.mutation_potentials import (
    DEFAULT_ALPHABET,
)
from pep_compass.steps.filters.strategies.ranked.scoring.latent_geometry.base import (
    MutationPoolScore,
)
from pep_compass.steps.filters.strategies.ranked.scoring.mutation_pool import (
    enumerate_sequences,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np

    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.autoencoder.subriemannian import SubRiemannianTangentSpace


class MoveScore(MutationPoolScore):
    """Score combinations by negative predicted net latent displacement."""

    def __init__(
        self,
        autoencoder: Autoencoder | None = None,
        maximum_candidates: int = 30000,
        alphabet: Sequence[str] | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder | None
        :param maximum_candidates: Maximum materialized candidates allowed by this local strategy.
        :type maximum_candidates: int
        :param alphabet: Ordered residue alphabet.
        :type alphabet: Sequence[str] | None
        :rtype: None
        """
        self._configure(
            {"maximum_candidates": maximum_candidates, "alphabet": alphabet}
        )
        self.autoencoder, self.maximum_candidates = autoencoder, maximum_candidates
        self.alphabet = alphabet or DEFAULT_ALPHABET

    @torch.no_grad()
    def score_group(
        self,
        parent: str,
        mutations: Mapping[int, Sequence[int]],
        tangent_space: SubRiemannianTangentSpace | None,
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        """Score group.

        :param parent: Parent.
        :type parent: str
        :param mutations: Mutations.
        :type mutations: Mapping[int, Sequence[int]]
        :param tangent_space: Tangent space.
        :type tangent_space: SubRiemannianTangentSpace | None
        :param rng: Run-owned random generator used for reproducible choices.
        :type rng: np.random.Generator | None
        :return: Result of this operation.
        :rtype: dict[str, float]
        """
        del tangent_space
        sequences = enumerate_sequences(
            parent, mutations, self.alphabet, self.maximum_candidates, rng=rng
        )
        if not sequences:
            return {}
        parent_latent = self.autoencoder.encode_peptides([parent])[0]  # (D,)
        keys = sorted(
            {
                (position, sequence[position])
                for sequence in sequences
                for position in range(len(parent))
                if sequence[position] != parent[position]
            }
        )
        singles = []
        for position, residue in keys:
            sequence = list(parent)
            sequence[position] = residue
            singles.append("".join(sequence))
        displacements = (
            self.autoencoder.encode_peptides(singles) - parent_latent
        )  # (M, D)
        lookup = {key: index for index, key in enumerate(keys)}
        scores = []
        for sequence in sequences:
            indices = [
                lookup[(position, sequence[position])]
                for position in range(len(parent))
                if sequence[position] != parent[position]
            ]
            net = displacements[indices].sum(dim=0)  # (D,)
            scores.append(float(-torch.linalg.vector_norm(net)))
        return dict(zip(sequences, scores))
