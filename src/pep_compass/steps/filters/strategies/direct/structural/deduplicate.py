"""Explicit candidate deduplication filter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.data_schemas.candidates import CandidateBatch
from pep_compass.steps.base import Step

if TYPE_CHECKING:
    from pep_compass.optimization.engine.execution.context import OptimizationContext


class Deduplicate(Step):
    """Keep the first candidate for every configured identity key."""

    category = "filter"
    display_parameters = ("key",)

    def __init__(self, key: str = "sequence") -> None:
        """Initialize this object with the declared configuration.

        :param key: Key.
        :type key: str
        :rtype: None
        """
        self._configure({"key": key})
        if key not in {"sequence", "sequence_and_latent"}:
            raise ValueError(
                "Deduplication key must be sequence or sequence_and_latent."
            )
        self.key = key

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
