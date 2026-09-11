"""Direct random controls for Local Enumeration experiments."""

from collections import defaultdict

import torch

from pep_compass.data.optimization import ObjectField
from pep_compass.optimization.components.filters.direct.base import DirectFilter
from pep_compass.optimization.components.filters.ranked.scoring.mutation_pool import enumerate_sequences
from pep_compass.optimization.components.filters.ranked.selection.nucleus import NucleusSelection
from pep_compass.optimization.components.filters.ranked.scoring.helpers.mutation_potentials import DEFAULT_ALPHABET


class RandomLeBoFilter(DirectFilter):
    """Apply random proposals or random ranking to each MUTANG parent pool."""

    def __init__(self, mode="walker", selection_fraction=0.6, temperature=1.0,
                 maximum_positions=5, residues_per_position=4,
                 maximum_candidates=30000, alphabet=None):
        if mode not in {"walker", "mutang_random"}:
            raise ValueError("Unknown random LE-BO mode.")
        self.mode, self.maximum_positions = mode, maximum_positions
        self.residues_per_position, self.maximum_candidates = residues_per_position, maximum_candidates
        self.alphabet = alphabet or DEFAULT_ALPHABET
        self.selection = NucleusSelection(selection_fraction, temperature)

    def _execute(self, batch, context):
        parents = batch.fields.get("mutation.parent_sequence")
        options = batch.fields.get("mutation.options")
        if not isinstance(parents, ObjectField) or not isinstance(options, ObjectField):
            raise ValueError("Random LE-BO requires MUTANG parent metadata.")
        groups = defaultdict(list)
        for index, (parent, mutation_map) in enumerate(zip(parents.values, options.values)):
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
                        index for index, token in enumerate(self.alphabet[1:], start=1)
                        if token != parent[position]
                    ]
                    mutations[position] = context.rng.choice(
                        residues, size=min(self.residues_per_position, len(residues)),
                        replace=False,
                    ).tolist()
            generated = enumerate_sequences(parent, mutations, self.alphabet, self.maximum_candidates)
            accepted = set(generated)
            if self.mode == "mutang_random" and generated:
                random_scores = torch.as_tensor(
                    context.rng.random(len(generated)), device=batch.latent_origins.device,
                    dtype=batch.latent_origins.dtype,
                )
                selected = self.selection.select(
                    random_scores, higher_is_better=True, context=context
                )
                accepted = {generated[index] for index in selected.tolist()}
            retained.extend(index for index in indices if batch.sequences[index] in accepted)
        return batch.select(torch.as_tensor(retained, device=batch.latent_origins.device))
