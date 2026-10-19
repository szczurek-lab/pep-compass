"""Parent-conditioned decoder likelihood scoring used by LPBEBO."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pep_compass.steps.filters.strategies.ranked.scoring.helpers.mutation_potentials import (
    DEFAULT_ALPHABET,
    DecoderLogProbabilityPotential,
    compose_mutant_distribution,
)
from pep_compass.steps.filters.strategies.ranked.scoring.latent_geometry.base import (
    MutationPoolScore,
)
from pep_compass.steps.filters.strategies.ranked.scoring.mutation_pool import (
    bounded_mutations,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np

    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.autoencoder.subriemannian import SubRiemannianTangentSpace


class DecoderLikelihoodScore(MutationPoolScore):
    """Score complete MUTANG combinations by summed decoder log-probability."""

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
        self.maximum_candidates = maximum_candidates
        self.alphabet = alphabet or DEFAULT_ALPHABET
        self.potential = DecoderLogProbabilityPotential(autoencoder, self.alphabet)

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
        bounded = bounded_mutations(
            parent, mutations, self.alphabet, self.maximum_candidates, rng=rng
        )
        distribution = compose_mutant_distribution(
            parent,
            bounded,
            self.potential,
            alphabet=self.alphabet,
            maximum_candidates=self.maximum_candidates,
        )
        return dict(zip(distribution.sequences, distribution.log_potentials))
