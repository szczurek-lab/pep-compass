"""Adapters for scores defined over candidates from one MUTANG parent."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from pep_compass.data_schemas.candidates import ObjectField, OptionalField
from pep_compass.steps.filters.strategies.ranked.scoring.base import (
    ScoreFunction,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np

    from pep_compass.autoencoder.subriemannian import SubRiemannianTangentSpace
    from pep_compass.data_schemas.candidates import CandidateBatch
    from pep_compass.data_schemas.contracts import StepContract
    from pep_compass.optimization.engine.execution.context import OptimizationContext


class MutationPoolScore(ScoreFunction):
    """Map a mutation-pool scoring implementation onto an aligned batch."""

    def contract(self) -> StepContract:
        """Require the MUTANG parent and options used for grouping candidate scores.

        :return: Result of this operation.
        :rtype: StepContract
        """
        from pep_compass.data_schemas.contracts import StepContract

        return StepContract(
            required_fields=("mutation.parent_sequence", "mutation.options"),
            optional_fields=("mutation.parent_sequence", "mutation.options"),
            resources=("autoencoder",),
        )

    def __call__(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> torch.Tensor:
        """Call.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
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
        for index, (parent, mutation_map) in enumerate(
            zip(parents.values, options.values)
        ):
            if valid is not None and not bool(valid[index]):
                continue
            groups[(parent, id(mutation_map))].append(index)
        # Parent identity is emitted by MUTANG but is intentionally absent from
        # scored mutant distributions. Negative infinity keeps it outside every
        # ranked selection without introducing a second structural filter.
        scores = torch.full(
            (len(batch),),
            float("-inf"),
            device=batch.latent_origins.device,
            dtype=batch.latent_origins.dtype,
        )
        for indices in groups.values():
            first = indices[0]
            tangent = (
                tangent_spaces.values[first]
                if isinstance(tangent_spaces, ObjectField)
                else None
            )
            score_by_sequence = self.score_group(
                parents.values[first], options.values[first], tangent, rng=context.rng
            )
            for index in indices:
                if batch.sequences[index] in score_by_sequence:
                    scores[index] = score_by_sequence[batch.sequences[index]]
        return scores

    def score_group(
        self,
        parent: str,
        mutations: Mapping[int, Sequence[int]],
        tangent_space: SubRiemannianTangentSpace | None,
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        """Return complete candidate sequences mapped to scalar scores.

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
        raise NotImplementedError
