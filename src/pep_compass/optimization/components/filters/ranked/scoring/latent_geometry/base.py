"""Adapters for scores defined over candidates from one MUTANG parent."""

from collections import defaultdict

import torch

from pep_compass.data.optimization import ObjectField, OptionalField
from pep_compass.optimization.components.filters.ranked.scoring.base import ScoreFunction


class MutationPoolScore(ScoreFunction):
    """Map a mutation-pool scoring implementation onto an aligned batch."""

    def __call__(self, batch, context):
        del context
        parents = batch.fields.get("mutation.parent_sequence")
        options = batch.fields.get("mutation.options")
        tangent_spaces = batch.fields.get("walker.tangent_space")
        # Local enumeration may mix MUTANG candidates with plain walk points
        # (``include_walk_points``); walk points carry no parent metadata and
        # CandidateBatch.concatenate surfaces that as an OptionalField once
        # merged with MUTANG output. Unwrap it and treat invalid rows the same
        # as unmatched sequences below: they stay at the -inf default instead
        # of aborting scoring for the whole batch.
        valid = None
        if isinstance(parents, OptionalField):
            valid = parents.valid
            parents = parents.values
        if isinstance(options, OptionalField):
            options_valid = options.valid
            options = options.values
            valid = options_valid if valid is None else valid & options_valid
        if not isinstance(parents, ObjectField) or not isinstance(options, ObjectField):
            raise ValueError("Latent-geometry scoring requires MUTANG parent metadata.")
        groups = defaultdict(list)
        for index, (parent, mutation_map) in enumerate(zip(parents.values, options.values)):
            if valid is not None and not bool(valid[index]):
                continue
            groups[(parent, id(mutation_map))].append(index)
        # Parent identity is emitted by MUTANG but is intentionally absent from
        # scored mutant distributions. Negative infinity keeps it outside every
        # ranked selection without introducing a second structural filter.
        scores = torch.full(
            (len(batch),), float("-inf"), device=batch.latent_origins.device,
            dtype=batch.latent_origins.dtype,
        )
        for indices in groups.values():
            first = indices[0]
            tangent = tangent_spaces.values[first] if isinstance(tangent_spaces, ObjectField) else None
            score_by_sequence = self.score_group(
                parents.values[first], options.values[first], tangent
            )
            for index in indices:
                if batch.sequences[index] in score_by_sequence:
                    scores[index] = score_by_sequence[batch.sequences[index]]
        return scores

    def score_group(self, parent, mutations, tangent_space):
        """Return complete candidate sequences mapped to scalar scores."""
        raise NotImplementedError
