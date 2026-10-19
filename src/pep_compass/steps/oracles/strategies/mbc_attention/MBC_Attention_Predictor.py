from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import torch


# Add local tools to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "tools"))

from config import def_bias, def_fts, def_scale
from tools.MultiBranchCNN import CNNimportFtsDataSetsPoNe, CNNstandardInputOutput

from pep_compass.steps.oracles.strategies.mbc_attention.artifacts import ARTIFACTS


class PredictorMBCAttention:
    """
    MBC Attention predictor following the same interface as APEX and BattleAMP predictors.
    Predicts antimicrobial activity (MIC values) for peptide sequences using Multi-Branch CNN.
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        batch_size: int = 3000,
        model: str = "default",
        models_directory: str | Path | None = None,
    ) -> None:
        """Initialize the MBC Attention predictor.

        Args:
            device (str): Device to use for computation ("cpu" or "cuda").
                         Note: TensorFlow will handle GPU allocation automatically.
            batch_size (int): Batch size for processing sequences.
            model (str): Registered MBC-Attention model variant.
            models_directory (str, optional): Replacement strategy package root.

        :param device: Computation device accepted by the numerical backend.
        :type device: str | torch.device
        :param batch_size: Batch size.
        :type batch_size: int
        :param model: Named model variant.
        :type model: str
        :param models_directory: Optional external model root replacing packaged artifacts.
        :type models_directory: str | Path | None
        :rtype: None
        """
        self.device = device
        self.batch_size = batch_size
        self.model_name = model

        # Model resolution
        ## Validate the SavedModel structure before TensorFlow initialization
        model_paths = ARTIFACTS.resolve(
            model,
            root=models_directory,
        )
        model_path = model_paths[0]

        # Configure TensorFlow to use GPU if available and requested
        if device == "cuda":
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                try:
                    # Enable memory growth to avoid allocating all GPU memory at once
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
            else:
                print("CUDA requested but no GPUs available, using CPU")

        self.model = keras.models.load_model(model_path)

        # Set pathogen list to match APEX structure (single model predicting general antimicrobial activity)
        self.pathogen_list = ["E. coli"]

        # Store feature configuration
        self.ft_list = def_fts
        self.bias = def_bias
        self.scale = def_scale

    def predict(self, seq_list: Sequence[str], use_tqdm: bool = False) -> np.ndarray:
        """Predict antimicrobial activity (MIC values) for a list of peptide sequences.

        Args:
            seq_list (list): List of peptide sequences as strings.
            use_tqdm (bool): Whether to show progress bars.

        Returns:
            np.ndarray: Predicted MIC values with shape (N, 1) where N is the number of sequences.
                       Values are in μM units (normal MIC, not log10).

        :param seq_list: Seq list.
        :type seq_list: Sequence[str]
        :param use_tqdm: Use tqdm.
        :type use_tqdm: bool
        :return: Result of this operation.
        :rtype: np.ndarray
        """
        data_len = len(seq_list)

        # Progress bar setup
        pbar = tqdm(total=data_len, desc="Processing sequences") if use_tqdm else None

        # Process sequences in batches
        batch_iter = range(int(math.ceil(data_len / float(self.batch_size))))
        all_predictions = []

        for i in batch_iter:
            # Get batch of sequences
            seq_batch = seq_list[i * self.batch_size : (i + 1) * self.batch_size]

            # Create fastas DataFrame directly from peptide list (no temp files)
            peptide_ids = [f"peptide_{j}" for j in range(len(seq_batch))]
            fastas = pd.DataFrame({"ID": peptide_ids, "SEQUENCE": seq_batch})

            # Generate CNN input datasets directly from DataFrame
            sets = CNNimportFtsDataSetsPoNe(fastas, ft_list=self.ft_list, target=None)

            # Prepare input for CNN
            X, Y = CNNstandardInputOutput(sets)

            # Make predictions
            batch_predictions = self.model.predict(X)

            # Convert predictions from scaled values back to log10 MIC values
            pred_log10_mic = batch_predictions / self.scale - self.bias

            # Convert from log10 MIC to normal MIC values (10^(-log10_mic))
            # Note: The model outputs -log10(MIC), so we need to negate and then take 10^x
            pred_mic = 10 ** (-pred_log10_mic)

            all_predictions.append(pred_mic)

            # Update progress bar
            if pbar is not None:
                pbar.update(len(seq_batch))

        if pbar is not None:
            pbar.close()

        # Concatenate all batch predictions
        predictions = np.vstack(all_predictions)

        return predictions
