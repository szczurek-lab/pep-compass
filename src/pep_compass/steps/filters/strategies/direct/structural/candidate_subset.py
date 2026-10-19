"""Fixed-size random and score-ranked candidate subset selection."""

from __future__ import annotations

from typing import Literal

import torch

from pep_compass.data_schemas.candidates import CandidateBatch, TensorField
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.steps.base import Step
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

SelectionMode = Literal["random", "highest", "lowest"]


class CandidateSubset(Step):
    """Select at most a fixed number of candidates from the complete batch.

    :param count: Maximum number of retained candidates.
    :type count: int
    :param mode: Uniform random selection or stable score ranking.
    :type mode: Literal["random", "highest", "lowest"]
    :param score_field: Candidate-aligned tensor field used by ranked modes.
    :type score_field: str | None
    :raises ValueError: If the count, mode, or score configuration is invalid.
    """

    category = "filter"
    input_fields = staticmethod(
        lambda parameters: (
            (parameters["score_field"],) if parameters["mode"] != "random" else ()
        )
    )
    output_bound = staticmethod(
        lambda parameters, count: min(count, parameters["count"])
    )
    display_parameters = ("count", "mode", "score_field")

    def __init__(
        self,
        *,
        count: int,
        mode: SelectionMode = "random",
        score_field: str | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param count: Count.
        :type count: int
        :param mode: Mode.
        :type mode: SelectionMode
        :param score_field: Score field.
        :type score_field: str | None
        :rtype: None
        """
        self._configure({"count": count, "mode": mode, "score_field": score_field})
        from pep_compass.utils.validation import validate_integer

        validate_integer(count, "CandidateSubset.count", 1)
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
        """Select the configured subset while retaining field alignment.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
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
        """Return stable ranked indices, excluding candidates with NaN scores.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param count: Count.
        :type count: int
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
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
