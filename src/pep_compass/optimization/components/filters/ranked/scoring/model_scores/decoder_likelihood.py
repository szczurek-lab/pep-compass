"""Parent-conditioned decoder likelihood scoring used by LPBEBO."""

from pep_compass.optimization.components.filters.ranked.scoring.latent_geometry.base import MutationPoolScore
from pep_compass.optimization.components.filters.ranked.scoring.mutation_pool import bounded_mutations
from pep_compass.optimization.components.filters.ranked.scoring.helpers.mutation_potentials import DEFAULT_ALPHABET, DecoderLogProbabilityPotential, compose_mutant_distribution


class DecoderLikelihoodScore(MutationPoolScore):
    """Score complete MUTANG combinations by summed decoder log-probability."""

    def __init__(self, autoencoder, maximum_candidates=30000, alphabet=None):
        self.maximum_candidates = maximum_candidates
        self.alphabet = alphabet or DEFAULT_ALPHABET
        self.potential = DecoderLogProbabilityPotential(autoencoder, self.alphabet)

    def score_group(self, parent, mutations, tangent_space):
        del tangent_space
        bounded = bounded_mutations(parent, mutations, self.alphabet, self.maximum_candidates)
        distribution = compose_mutant_distribution(
            parent, bounded, self.potential, alphabet=self.alphabet,
            maximum_candidates=self.maximum_candidates,
        )
        return dict(zip(distribution.sequences, distribution.log_potentials))
