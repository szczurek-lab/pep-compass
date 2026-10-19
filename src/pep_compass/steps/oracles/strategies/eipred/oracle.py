from __future__ import annotations

import numpy as np
from poli.core.black_box_information import BlackBoxInformation
from poli_baselines.core.abstract_solver import AbstractBlackBox

from pep_compass.steps.oracles.strategies.eipred.eippred import EIPredPredictor


class EIPredBlackBox(AbstractBlackBox):
    def __init__(
        self,
        *,
        mic_aggregate: str = "mean",
        batch_size: int | None = None,
        parallelize: bool = False,
        num_workers: int | None = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
        model: str = "default",
        models_directory: str | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param mic_aggregate: Mic aggregate.
        :type mic_aggregate: str
        :param batch_size: Batch size.
        :type batch_size: int | None
        :param parallelize: Parallelize.
        :type parallelize: bool
        :param num_workers: Num workers.
        :type num_workers: int | None
        :param evaluation_budget: Evaluation budget.
        :type evaluation_budget: int
        :param force_isolation: Force isolation.
        :type force_isolation: bool
        :param model: Named model variant.
        :type model: str
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

        # Model-backed predictor
        ## Resolve both classifier weights and the selected-feature schema
        self.predictor = EIPredPredictor(
            device="cpu",
            batch_size=batch_size or 1000,
            model=model,
            models_directory=models_directory,
        )

        if mic_aggregate == "max":
            mic_aggregate_func = lambda x: np.max(x, axis=1)
        elif mic_aggregate == "mean":
            mic_aggregate_func = lambda x: np.mean(x, axis=1)
        else:
            raise ValueError("mic_aggregate must be 'mean' or 'max'")

        # EIPred model outputs 10**(-y_pred), apply log2 transformation
        self.peptide_scorer = lambda seqs: mic_aggregate_func(
            np.log2(self.predictor.predict(seqs)).reshape(-1, 1)
        )

        self.cache = []
        self.shift = 0.0

    def set_shift(self, shift: float) -> None:
        """Set shift.

        :param shift: Shift.
        :type shift: float
        :rtype: None
        """
        self.shift = shift

    def get_black_box_info(self) -> BlackBoxInformation:
        """Get black box info.

        :return: Result of this operation.
        :rtype: BlackBoxInformation
        """
        return BlackBoxInformation(
            name="EIPred",
            max_sequence_length=50,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=list("ACDEFGHIKLMNPQRSTVWY"),
            log_transform_recommended=False,
            discrete=True,
            padding_token=" ",
        )

    def _black_box(self, x: np.ndarray, context: dict | None = None) -> np.ndarray:
        # x expected as array-like of sequences (list of chars per sequence)
        """Black box.

        :param x: X.
        :type x: np.ndarray
        :param context: Scoped execution services, shared run state and random generator.
        :type context: dict | None
        :return: Result of this operation.
        :rtype: np.ndarray
        """
        sequences = ["".join(seq) for seq in x]
        predictions = self.peptide_scorer(sequences)

        if context is not None and isinstance(context, dict):
            context["sequences"] = sequences

        for i, seq in enumerate(sequences):
            self.cache.append((seq, float(predictions[i])))

        return predictions.reshape(-1, 1) + self.shift

    def clear_cache(self) -> None:
        """Clear cache.

        :rtype: None
        """
        self.cache = []
