"""Explicit candidate deduplication filter."""

import torch

from pep_compass.data.optimization import CandidateBatch
from pep_compass.optimization.components.filters.direct.base import DirectFilter
from pep_compass.optimization.components.filters.manager import FilterManager


@FilterManager.register("deduplicate")
class DeduplicateFilter(DirectFilter):
    """Keep the first candidate for every configured identity key."""

    def __init__(self, key: str = "sequence") -> None:
        if key not in {"sequence", "sequence_and_latent"}:
            raise ValueError("Deduplication key must be sequence or sequence_and_latent.")
        self.key = key

    def _execute(self, batch: CandidateBatch, context) -> CandidateBatch:
        del context
        seen, retained = set(), []
        latent_rows = batch.latent_origins.detach().cpu()
        for index, sequence in enumerate(batch.sequences):
            identity = sequence
            if self.key == "sequence_and_latent":
                identity = (sequence, latent_rows[index].numpy().tobytes())
            if identity not in seen:
                seen.add(identity)
                retained.append(index)
        indices = torch.as_tensor(retained, device=batch.latent_origins.device)
        return batch.select(indices)
