"""Shared operations for scoring complete MUTANG candidate pools."""

from __future__ import annotations

import itertools
import math
import random
from typing import TYPE_CHECKING

from pep_compass.autoencoder.subriemannian import SubRiemannianTangentSpace

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np

    from pep_compass.autoencoder.base import Autoencoder


def bounded_mutations(
    parent: str,
    mutations: Mapping[int, Sequence[int]],
    alphabet: Sequence[str],
    maximum_candidates: int,
    *,
    rng: np.random.Generator | None = None,
) -> dict[int, list[int]]:
    """Reduce a Cartesian mutation product before it is materialised.

    :param parent: Parent.
    :type parent: str
    :param mutations: Mutations.
    :type mutations: Mapping[int, Sequence[int]]
    :param alphabet: Ordered residue alphabet.
    :type alphabet: Sequence[str]
    :param maximum_candidates: Maximum materialized candidates allowed by this local strategy.
    :type maximum_candidates: int
    :param rng: Run-owned random generator used for reproducible choices.
    :type rng: np.random.Generator | None
    :return: Result of this operation.
    :rtype: dict[int, list[int]]
    """
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
            choices[position].remove(
                (
                    random.choice(alternatives)
                    if rng is None
                    else int(rng.choice(alternatives))
                )
            )
        elif len(choices) > 1:
            del choices[position]
        else:
            break
    return choices


def enumerate_sequences(
    parent: str,
    mutations: Mapping[int, Sequence[int]],
    alphabet: Sequence[str],
    maximum_candidates: int,
    *,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Return the bounded unique product excluding the unchanged parent.

    :param parent: Parent.
    :type parent: str
    :param mutations: Mutations.
    :type mutations: Mapping[int, Sequence[int]]
    :param alphabet: Ordered residue alphabet.
    :type alphabet: Sequence[str]
    :param maximum_candidates: Maximum materialized candidates allowed by this local strategy.
    :type maximum_candidates: int
    :param rng: Run-owned random generator used for reproducible choices.
    :type rng: np.random.Generator | None
    :return: Result of this operation.
    :rtype: list[str]
    """
    choices = bounded_mutations(
        parent, mutations, alphabet, maximum_candidates, rng=rng
    )
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


def tangent_space_for_parent(
    autoencoder: Autoencoder | None,
    parent: str,
    supplied: SubRiemannianTangentSpace | None,
    threshold: float,
) -> SubRiemannianTangentSpace:
    """Use shared point geometry or compute standalone parent geometry once.

    :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
    :type autoencoder: Autoencoder | None
    :param parent: Parent.
    :type parent: str
    :param supplied: Supplied.
    :type supplied: SubRiemannianTangentSpace | None
    :param threshold: Threshold.
    :type threshold: float
    :return: Result of this operation.
    :rtype: SubRiemannianTangentSpace
    """
    if supplied is not None:
        return supplied
    latent = autoencoder.encode_peptides([parent])
    decomposition = autoencoder.decoder_jacobian(latent)[0]
    import torch

    left, singular, right = torch.linalg.svd(decomposition, full_matrices=False)
    return SubRiemannianTangentSpace(left, singular, right, threshold)
