"""Clean POLI-compatible APEX oracle strategy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation

from pep_compass.steps.oracles.strategies.apex.predictor import (
    APEXPredictor,
)
from pep_compass.steps.oracles.strategies.apex.validation import (
    validate_bacteria_indices,
)


class APEXBlackBox(AbstractBlackBox):
    """Aggregate pathogen-specific APEX MIC predictions into one objective.

    The public APEX component prepares this numerical implementation and then
    wraps it with the common ``BlackBoxOracle`` step. Consequently, budget
    accounting and attachment of ``oracle.apex.score`` occur in the shared
    oracle adapter rather than in this model-specific implementation.

    :param mic_aggregate: ``mean`` or ``max`` reduction over pathogen columns.
    :param mic_bacteria: ``all`` or zero-based pathogen column indices.
    :param model: Named ensemble name.
    :param device: Torch inference device.
    :param models_directory: Optional external root containing named ensembles.
    """

    def __init__(
        self,
        *,
        mic_aggregate: str = "mean",
        mic_bacteria: str | Sequence[int] = "all",
        model: str = "default",
        batch_size: int | None = None,
        parallelize: bool = False,
        num_workers: int | None = None,
        evaluation_budget: int | float = float("inf"),
        force_isolation: bool = False,
        device: str = "cpu",
        models_directory: str | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param mic_aggregate: Mic aggregate.
        :type mic_aggregate: str
        :param mic_bacteria: Mic bacteria.
        :type mic_bacteria: str | Sequence[int]
        :param model: Named model variant.
        :type model: str
        :param batch_size: Batch size.
        :type batch_size: int | None
        :param parallelize: Parallelize.
        :type parallelize: bool
        :param num_workers: Num workers.
        :type num_workers: int | None
        :param evaluation_budget: Evaluation budget.
        :type evaluation_budget: int | float
        :param force_isolation: Force isolation.
        :type force_isolation: bool
        :param device: Computation device accepted by the numerical backend.
        :type device: str
        :param models_directory: Optional external model root replacing packaged artifacts.
        :type models_directory: str | None
        :rtype: None
        """
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        if mic_aggregate not in {"mean", "max"}:
            raise ValueError("APEX mic_aggregate must be 'mean' or 'max'.")
        self.mic_aggregate = mic_aggregate
        self.predictor = APEXPredictor(
            device=device,
            model=model,
            models_directory=models_directory,
        )
        self.bacteria_indices = validate_bacteria_indices(
            mic_bacteria,
            pathogen_count=len(self.predictor.pathogen_list),
        )
        self.maximize = False

    def score(self, sequences: Sequence[str]) -> np.ndarray:
        """Return aggregated log2 MIC values shaped ``(B,)``.

        :param sequences: Peptide sequences in input order.
        :type sequences: Sequence[str]
        :return: Result of this operation.
        :rtype: np.ndarray
        """
        predictions = self.predictor.predict(sequences)  # (B, P)
        if self.bacteria_indices is not None:
            predictions = predictions[:, self.bacteria_indices]  # (B, P_selected)
        if np.any(predictions <= 0) or not np.all(np.isfinite(predictions)):
            raise ValueError("APEX produced non-positive or non-finite MIC values.")
        transformed = np.log2(predictions)  # (B, P_selected)
        reduction = np.mean if self.mic_aggregate == "mean" else np.max
        return reduction(transformed, axis=1)  # (B,)

    def get_black_box_info(self) -> BlackBoxInformation:
        """Return the static POLI declaration for APEX peptide inputs.

        :return: Result of this operation.
        :rtype: BlackBoxInformation
        """
        return BlackBoxInformation(
            name="APEX",
            max_sequence_length=25,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=list("ACDEFGHIKLMNPQRSTVWY"),
            log_transform_recommended=False,
            discrete=True,
            padding_token=" ",
        )

    def _black_box(self, x: np.ndarray, context: dict | None = None) -> np.ndarray:
        """Translate POLI token rows and return scores shaped ``(B, 1)``.

        :param x: X.
        :type x: np.ndarray
        :param context: Scoped execution services, shared run state and random generator.
        :type context: dict | None
        :return: Result of this operation.
        :rtype: np.ndarray
        """
        sequences = ["".join(row) for row in x]
        return self.score(sequences).reshape(-1, 1)  # (B, 1)
