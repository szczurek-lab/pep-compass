"""MOVE candidate scoring."""

import torch

from pep_compass.optimization.components.filters.ranked.scoring.latent_geometry.base import MutationPoolScore
from pep_compass.optimization.components.filters.ranked.scoring.mutation_pool import enumerate_sequences
from pep_compass.optimization.components.filters.ranked.scoring.helpers.mutation_potentials import DEFAULT_ALPHABET


class MoveScore(MutationPoolScore):
    """Score combinations by negative predicted net latent displacement."""

    def __init__(self, autoencoder, maximum_candidates=30000, alphabet=None):
        self.autoencoder, self.maximum_candidates = autoencoder, maximum_candidates
        self.alphabet = alphabet or DEFAULT_ALPHABET

    @torch.no_grad()
    def score_group(self, parent, mutations, tangent_space):
        del tangent_space
        sequences = enumerate_sequences(
            parent, mutations, self.alphabet, self.maximum_candidates
        )
        if not sequences:
            return {}
        parent_latent = self.autoencoder.encode_peptides([parent])[0]  # (D,)
        keys = sorted({
            (position, sequence[position]) for sequence in sequences
            for position in range(len(parent)) if sequence[position] != parent[position]
        })
        singles = []
        for position, residue in keys:
            sequence = list(parent)
            sequence[position] = residue
            singles.append("".join(sequence))
        displacements = self.autoencoder.encode_peptides(singles) - parent_latent  # (M, D)
        lookup = {key: index for index, key in enumerate(keys)}
        scores = []
        for sequence in sequences:
            indices = [
                lookup[(position, sequence[position])] for position in range(len(parent))
                if sequence[position] != parent[position]
            ]
            net = displacements[indices].sum(dim=0)  # (D,)
            scores.append(float(-torch.linalg.vector_norm(net)))
        return dict(zip(sequences, scores))
