"""Fixed-size random and score-ranked candidate subset selection."""

from __future__ import annotations

from typing import Literal

import torch

from pep_compass.optimization.components.filters.direct.base import DirectFilter
from pep_compass.optimization.components.filters.manager import FilterManager
from pep_compass.data.optimization import CandidateBatch, TensorField
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

SelectionMode = Literal["random", "highest", "lowest"]


@FilterManager.register("candidate_subset")
class CandidateSubsetSelector(DirectFilter):
    """Select at most a fixed number of candidates from the complete batch.

    :param count: Maximum number of retained candidates.
    :type count: int
    :param mode: Uniform random selection or stable score ranking.
    :type mode: Literal["random", "highest", "lowest"]
    :param score_field: Candidate-aligned tensor field used by ranked modes.
    :type score_field: str | None
    :raises ValueError: If the count, mode, or score configuration is invalid.
    """

    def __init__(
        self,
        *,
        count: int,
        mode: SelectionMode = "random",
        score_field: str | None = None,
    ) -> None:
        if count < 1:
            raise ValueError("Candidate subset count must be positive.")
        if mode not in {"random", "highest", "lowest"}:
            raise ValueError(
                "Candidate subset mode must be random, highest, or lowest."
            )
        if mode != "random" and not score_field:
            raise ValueError("Ranked candidate subset modes require score_field.")
        self.count = count
        self.mode = mode
        self.score_field = score_field

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Select the configured subset while retaining field alignment."""
        count = min(self.count, len(batch))
        if count == len(batch):
            return batch
        if self.mode == "random":
            indices = context.rng.choice(len(batch), size=count, replace=False)
            selected = torch.as_tensor(
                indices,
                dtype=torch.long,
                device=batch.latent_origins.device,
            )
        else:
            selected = self._ranked_indices(batch, count)
        logger.debug(
            "Candidate subset mode=%s retained=%s input=%s score_field=%s.",
            self.mode,
            len(selected),
            len(batch),
            self.score_field,
        )
        return batch.select(selected)

    def _ranked_indices(
        self,
        batch: CandidateBatch,
        count: int,
    ) -> torch.Tensor:
        """Return stable ranked indices, excluding candidates with NaN scores."""
        score = batch.fields.get(self.score_field or "")
        if not isinstance(score, TensorField) or score.values.ndim != 1:
            raise ValueError(
                "Candidate subset score_field must reference a one-dimensional "
                "TensorField."
            )
        valid = torch.nonzero(~torch.isnan(score.values), as_tuple=False).flatten()
        valid_scores = score.values.index_select(0, valid)
        order = torch.argsort(
            valid_scores,
            descending=self.mode == "highest",
            stable=True,
        )
        return valid.index_select(0, order[:count])
