"""Shared operations for scoring complete MUTANG candidate pools."""

import itertools
import math
import random

from pep_compass.autoencoder.subriemannian import SubRiemannianTangentSpace


def bounded_mutations(parent, mutations, alphabet, maximum_candidates):
    """Reduce a Cartesian mutation product before it is materialised."""
    valid = {
        position: sorted(set(residues))
        for position, residues in mutations.items()
        if 0 <= position < len(parent) and residues
    }
    choices = {
        position: sorted(set(residues) | {alphabet.index(parent[position])})
        for position, residues in valid.items()
    }
    while choices and math.prod(map(len, choices.values())) > maximum_candidates:
        position = max(choices, key=lambda key: len(choices[key]))
        parent_residue = alphabet.index(parent[position])
        alternatives = [value for value in choices[position] if value != parent_residue]
        if alternatives:
            choices[position].remove(random.choice(alternatives))
        elif len(choices) > 1:
            del choices[position]
        else:
            break
    return choices


def enumerate_sequences(parent, mutations, alphabet, maximum_candidates):
    """Return the bounded unique product excluding the unchanged parent."""
    choices = bounded_mutations(parent, mutations, alphabet, maximum_candidates)
    positions = sorted(choices)
    sequences = []
    for residues in itertools.product(*(choices[position] for position in positions)):
        sequence = list(parent)
        for position, residue in zip(positions, residues):
            sequence[position] = alphabet[residue]
        candidate = "".join(sequence)
        if candidate != parent:
            sequences.append(candidate)
    return list(dict.fromkeys(sequences))


def tangent_space_for_parent(autoencoder, parent, supplied, threshold):
    """Use shared point geometry or compute standalone parent geometry once."""
    if supplied is not None:
        return supplied
    latent = autoencoder.encode_peptides([parent])
    decomposition = autoencoder.decoder_jacobian(latent)[0]
    import torch

    left, singular, right = torch.linalg.svd(decomposition, full_matrices=False)
    return SubRiemannianTangentSpace(left, singular, right, threshold)
