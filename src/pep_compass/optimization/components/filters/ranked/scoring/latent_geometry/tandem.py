"""TANDEM candidate scoring."""

from pep_compass.optimization.components.filters.ranked.scoring.latent_geometry.base import MutationPoolScore
from pep_compass.optimization.components.filters.ranked.scoring.mutation_pool import bounded_mutations, tangent_space_for_parent
from pep_compass.optimization.components.filters.ranked.scoring.helpers.mutation_potentials import DEFAULT_ALPHABET, ProjectedDirectionPairwiseSimilarityPotential, compose_mutant_distribution


class TandemScore(MutationPoolScore):
    """Score complete mutation combinations by projected pair similarity."""

    def __init__(self, autoencoder, horizontal_threshold=0.1, maximum_candidates=30000, alphabet=None):
        self.autoencoder, self.horizontal_threshold = autoencoder, horizontal_threshold
        self.maximum_candidates = maximum_candidates
        self.alphabet = alphabet or DEFAULT_ALPHABET

    def score_group(self, parent, mutations, tangent_space):
        tangent = tangent_space_for_parent(
            self.autoencoder, parent, tangent_space, self.horizontal_threshold
        )
        bounded = bounded_mutations(parent, mutations, self.alphabet, self.maximum_candidates)
        distribution = compose_mutant_distribution(
            parent, bounded,
            ProjectedDirectionPairwiseSimilarityPotential(tangent, self.alphabet),
            alphabet=self.alphabet, include_parent_residue=True,
            maximum_candidates=self.maximum_candidates,
        )
        return dict(zip(distribution.sequences, distribution.log_potentials))
