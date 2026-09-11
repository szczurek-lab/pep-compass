"""APEX sequence black box adapted to the common PepCompass oracle step."""

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation

from pep_compass.optimization.components.oracles.strategies.apex_original.predictor import PredictorAPEX


class APEXBlackBox(AbstractBlackBox):
    """Evaluate peptide sequences with an averaged pretrained APEX ensemble.

    ``OracleManager`` constructs this POLI-compatible implementation lazily and
    wraps it in :class:`BlackBoxOracle`, which supplies the PepCompass
    :class:`Oracle` lifecycle, budget accounting and result fields.

    :param mic_aggregate: Reduction across selected pathogen MIC columns.
    :param mic_bacteria: ``all`` or zero-based pathogen column indices.
    :param model: Registered APEX ensemble variant.
    :param device: Torch inference device.
    :param evaluation_budget: Optional POLI-side evaluation ceiling.
    :raises ValueError: If aggregation or pathogen selection is invalid.
    :raises FileNotFoundError: If the requested APEX weights are incomplete.
    """

    def __init__(
        self,
        *,
        mic_aggregate: str = "mean",
        mic_bacteria: str | list = "all",
        model: str = "default",
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
        device: str = "cpu",
        models_directory: str | None = None,
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )

        if mic_aggregate == "max":
            self._aggregate = lambda values: np.max(values, axis=1)
        elif mic_aggregate == "mean":
            self._aggregate = lambda values: np.mean(values, axis=1)
        else:
            raise ValueError("APEX mic_aggregate must be 'mean' or 'max'.")

        if mic_bacteria == "all":
            self._bacteria_indices: tuple[int, ...] | None = None
        elif isinstance(mic_bacteria, list):
            if not mic_bacteria or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                for index in mic_bacteria
            ):
                raise ValueError(
                    "APEX mic_bacteria must contain valid zero-based pathogen indices."
                )
            self._bacteria_indices = tuple(mic_bacteria)
        else:
            raise ValueError("APEX mic_bacteria must be 'all' or a list of indices.")

        self.apex_predictor = PredictorAPEX(
            device=device,
            model=model,
            models_directory=models_directory,
        )
        if self._bacteria_indices is not None and any(
            index >= len(self.apex_predictor.pathogen_list)
            for index in self._bacteria_indices
        ):
            raise ValueError(
                "APEX mic_bacteria contains an index outside the selected model."
            )

        self.maximize = False

    def score(self, sequences: list[str]) -> np.ndarray:
        """Return aggregated log2 MIC scores shaped ``(B,)``."""
        predictions = self.apex_predictor.predict(sequences)  # (B, P)
        if self._bacteria_indices is not None:
            predictions = predictions[:, self._bacteria_indices]  # (B, P_selected)
        if np.any(predictions <= 0) or not np.all(np.isfinite(predictions)):
            raise ValueError("APEX produced non-positive or non-finite MIC values.")
        return self._aggregate(np.log2(predictions))  # (B,)

    def get_black_box_info(self) -> BlackBoxInformation:
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

    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        """Translate POLI token rows and return scores shaped ``(B, 1)``."""
        sequences = ["".join(seq) for seq in x]
        predictions = self.score(sequences)

        return predictions.reshape(-1, 1)
