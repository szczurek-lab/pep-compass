"""Direct random controls for Local Enumeration experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from pep_compass.data_schemas.candidates import ObjectField
from pep_compass.steps.base import Step
from pep_compass.steps.filters.strategies.ranked.scoring.helpers.mutation_potentials import (
    DEFAULT_ALPHABET,
)
from pep_compass.steps.filters.strategies.ranked.scoring.mutation_pool import (
    enumerate_sequences,
)
from pep_compass.steps.filters.strategies.ranked.selection.nucleus import (
    NucleusSelection,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pep_compass.data_schemas.candidates import CandidateBatch
    from pep_compass.optimization.engine.execution.context import OptimizationContext


class RandomMutationPool(Step):
    """Apply random proposals or random ranking to each MUTANG parent pool."""

    category = "filter"

    def __init__(
        self,
        mode: str = "walker",
        selection_fraction: float = 0.6,
        temperature: float = 1.0,
        maximum_positions: int = 5,
        residues_per_position: int = 4,
        maximum_candidates: int = 30000,
        alphabet: Sequence[str] | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param mode: Mode.
        :type mode: str
        :param selection_fraction: Selection fraction.
        :type selection_fraction: float
        :param temperature: Temperature.
        :type temperature: float
        :param maximum_positions: Maximum positions.
        :type maximum_positions: int
        :param residues_per_position: Residues per position.
        :type residues_per_position: int
        :param maximum_candidates: Maximum materialized candidates allowed by this local strategy.
        :type maximum_candidates: int
        :param alphabet: Ordered residue alphabet.
        :type alphabet: Sequence[str] | None
        :rtype: None
        """
        self._configure(
            {
                "mode": mode,
                "selection_fraction": selection_fraction,
                "temperature": temperature,
                "maximum_positions": maximum_positions,
                "residues_per_position": residues_per_position,
                "maximum_candidates": maximum_candidates,
                "alphabet": alphabet,
            }
        )
        if mode not in {"walker", "mutang_random"}:
            raise ValueError("Unknown random LE-BO mode.")
        self.mode, self.maximum_positions = mode, maximum_positions
        self.residues_per_position, self.maximum_candidates = (
            residues_per_position,
            maximum_candidates,
        )
        self.alphabet = alphabet or DEFAULT_ALPHABET
        self.selection = NucleusSelection(selection_fraction, temperature)

    def _execute(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> CandidateBatch:
        """Execute.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        parents = batch.fields.get("mutation.parent_sequence")
        options = batch.fields.get("mutation.options")
        if not isinstance(parents, ObjectField) or not isinstance(options, ObjectField):
            raise ValueError("Random LE-BO requires MUTANG parent metadata.")
        groups = defaultdict(list)
        for index, (parent, mutation_map) in enumerate(
            zip(parents.values, options.values)
        ):
            groups[(parent, id(mutation_map))].append(index)
        retained = []
        for indices in groups.values():
            first, parent = indices[0], parents.values[indices[0]]
            mutations = options.values[first]
            if self.mode == "walker":
                count = min(len(parent), self.maximum_positions)
                positions = context.rng.choice(len(parent), size=count, replace=False)
                mutations = {}
                for position in positions.tolist():
                    residues = [
                        index
                        for index, token in enumerate(self.alphabet[1:], start=1)
                        if token != parent[position]
                    ]
                    mutations[position] = context.rng.choice(
                        residues,
                        size=min(self.residues_per_position, len(residues)),
                        replace=False,
                    ).tolist()
            generated = enumerate_sequences(
                parent,
                mutations,
                self.alphabet,
                self.maximum_candidates,
                rng=context.rng,
            )
            accepted = set(generated)
            if self.mode == "mutang_random" and generated:
                random_scores = torch.as_tensor(
                    context.rng.random(len(generated)),
                    device=batch.latent_origins.device,
                    dtype=batch.latent_origins.dtype,
                )
                selected = self.selection.select(
                    random_scores, higher_is_better=True, context=context
                )
                accepted = {generated[index] for index in selected.tolist()}
            retained.extend(
                index for index in indices if batch.sequences[index] in accepted
            )
        return batch.select(
            torch.as_tensor(retained, device=batch.latent_origins.device)
        )
