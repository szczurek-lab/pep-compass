from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from poli.core.black_box_information import BlackBoxInformation
from poli_baselines.core.abstract_solver import AbstractBlackBox

from pep_compass.steps.oracles.strategies.mbc_attention.MBC_Attention_Predictor import (
    PredictorMBCAttention,
)


class MBCAttentionBlackBox(AbstractBlackBox):
    def __init__(
        self,
        *,
        batch_size: int | None = None,
        parallelize: bool = False,
        num_workers: int | None = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
        device: str = "cpu",
        model: str = "default",
        models_directory: str | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

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
        :param device: Computation device accepted by the numerical backend.
        :type device: str
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

        self.device = device
        self.mbc_attention_predictor = PredictorMBCAttention(
            device=device,
            model=model,
            models_directory=models_directory,
        )

        # MBC Attention returns a single prediction value, so no aggregation needed
        self.peptide_scorer = lambda x: np.log2(
            self.mbc_attention_predictor.predict(x).flatten()
        )

        self.cache = []

    def get_black_box_info(self) -> BlackBoxInformation:
        """Get black box info.

        :return: Result of this operation.
        :rtype: BlackBoxInformation
        """
        return BlackBoxInformation(
            name="MBCAttention",
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

        for i, seq in enumerate(sequences):
            self.cache.append((seq, predictions[i].item()))

        return predictions.reshape(-1, 1)

    def clear_cache(self) -> None:
        """Clear cache.

        :rtype: None
        """
        self.cache = []
