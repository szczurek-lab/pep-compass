"""LAMS candidate scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pep_compass.steps.filters.strategies.ranked.scoring.helpers.mutation_potentials import (
    DEFAULT_ALPHABET,
    LamsAnchorSimilarityPotential,
    ProjectedDirectionPairwiseSimilarityPotential,
    compose_mutant_distribution,
)
from pep_compass.steps.filters.strategies.ranked.scoring.latent_geometry.base import (
    MutationPoolScore,
)
from pep_compass.steps.filters.strategies.ranked.scoring.mutation_pool import (
    bounded_mutations,
    tangent_space_for_parent,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np

    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.autoencoder.subriemannian import SubRiemannianTangentSpace


class LamsScore(MutationPoolScore):
    """Score combinations by their worst compatible mutation pair."""

    def __init__(
        self,
        autoencoder: Autoencoder | None = None,
        horizontal_threshold: float = 0.1,
        maximum_candidates: int = 30000,
        alphabet: Sequence[str] | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder | None
        :param horizontal_threshold: Horizontal threshold.
        :type horizontal_threshold: float
        :param maximum_candidates: Maximum materialized candidates allowed by this local strategy.
        :type maximum_candidates: int
        :param alphabet: Ordered residue alphabet.
        :type alphabet: Sequence[str] | None
        :rtype: None
        """
        self._configure(
            {
                "horizontal_threshold": horizontal_threshold,
                "maximum_candidates": maximum_candidates,
                "alphabet": alphabet,
            }
        )
        self.autoencoder, self.horizontal_threshold = autoencoder, horizontal_threshold
        self.maximum_candidates = maximum_candidates
        self.alphabet = alphabet or DEFAULT_ALPHABET

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
        tangent = tangent_space_for_parent(
            self.autoencoder, parent, tangent_space, self.horizontal_threshold
        )
        bounded = bounded_mutations(
            parent, mutations, self.alphabet, self.maximum_candidates, rng=rng
        )
        potential = LamsAnchorSimilarityPotential(
            ProjectedDirectionPairwiseSimilarityPotential(tangent, self.alphabet)
        )
        distribution = compose_mutant_distribution(
            parent,
            bounded,
            potential,
            alphabet=self.alphabet,
            include_parent_residue=True,
            maximum_candidates=self.maximum_candidates,
        )
        return dict(zip(distribution.sequences, distribution.log_potentials))
