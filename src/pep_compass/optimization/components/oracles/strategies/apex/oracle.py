"""Clean POLI-compatible APEX oracle strategy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation

from pep_compass.optimization.components.oracles.strategies.apex.predictor import (
    APEXPredictor,
)


class APEXBlackBox(AbstractBlackBox):
    """Aggregate pathogen-specific APEX MIC predictions into one objective.

    The runtime lazily constructs this class through ``OracleManager`` and then
    wraps it with the common ``BlackBoxOracle`` step. Consequently, budget
    accounting and attachment of ``oracle.apex.score`` occur in the shared
    oracle adapter rather than in this model-specific implementation.

    :param mic_aggregate: ``mean`` or ``max`` reduction over pathogen columns.
    :param mic_bacteria: ``all`` or zero-based pathogen column indices.
    :param model: Registered ensemble name.
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
        self.bacteria_indices = _validate_bacteria_indices(
            mic_bacteria,
            pathogen_count=len(self.predictor.pathogen_list),
        )
        self.maximize = False

    def score(self, sequences: Sequence[str]) -> np.ndarray:
        """Return aggregated log2 MIC values shaped ``(B,)``."""
        predictions = self.predictor.predict(sequences)  # (B, P)
        if self.bacteria_indices is not None:
            predictions = predictions[:, self.bacteria_indices]  # (B, P_selected)
        if np.any(predictions <= 0) or not np.all(np.isfinite(predictions)):
            raise ValueError("APEX produced non-positive or non-finite MIC values.")
        transformed = np.log2(predictions)  # (B, P_selected)
        reduction = np.mean if self.mic_aggregate == "mean" else np.max
        return reduction(transformed, axis=1)  # (B,)

    def get_black_box_info(self) -> BlackBoxInformation:
        """Return the static POLI declaration for APEX peptide inputs."""
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
        """Translate POLI token rows and return scores shaped ``(B, 1)``."""
        sequences = ["".join(row) for row in x]
        return self.score(sequences).reshape(-1, 1)  # (B, 1)


def _validate_bacteria_indices(
    value: str | Sequence[int],
    *,
    pathogen_count: int,
) -> tuple[int, ...] | None:
    """Normalize and validate configured pathogen columns."""
    if value == "all":
        return None
    if isinstance(value, str) or not value:
        raise ValueError("APEX mic_bacteria must be 'all' or a non-empty index list.")
    indices = tuple(value)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < pathogen_count
        for index in indices
    ):
        raise ValueError("APEX mic_bacteria contains an invalid pathogen index.")
    return indices
