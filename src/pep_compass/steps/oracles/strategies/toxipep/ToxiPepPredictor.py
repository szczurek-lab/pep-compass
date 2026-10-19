"""
ToxiPepPredictor: Unified predictor interface for ToxiPep toxicity prediction model

This module provides a standardized interface for the ToxiPep model, which predicts
peptide toxicity using a hybrid Transformer and CNN architecture with molecular features.
The predictor follows the same interface pattern as other models in the pep-compass framework.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from pep_compass.steps.oracles.strategies.toxipep.artifacts import ARTIFACTS
from pep_compass.steps.oracles.strategies.toxipep.atom_feature import (
    convert_to_graph_channel,
)
from pep_compass.steps.oracles.strategies.toxipep.model import (
    ToxiPep_Model,
)

# Peptide residue mapping - same as in original ToxiPep
Pep_residue2idx = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "A": 3,
    "C": 4,
    "D": 5,
    "E": 6,
    "F": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "K": 11,
    "L": 12,
    "M": 13,
    "N": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "V": 20,
    "W": 21,
    "Y": 22,
}


class PeptideDataset(Dataset):
    """Dataset for peptide sequences with graph features"""

    def __init__(
        self, sequences: Sequence[str], graph_features: torch.Tensor | np.ndarray
    ) -> None:
        """Initialize this object with the declared configuration.

        :param sequences: Peptide sequences in input order.
        :type sequences: Sequence[str]
        :param graph_features: Graph features.
        :type graph_features: torch.Tensor | np.ndarray
        :rtype: None
        """
        self.sequences = sequences
        self.graph_features = graph_features

    def __len__(self) -> int:
        """Return the number of contained elements.

        :return: Result of this operation.
        :rtype: int
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the element selected by the requested index.

        :param idx: Idx.
        :type idx: int
        :return: Result of this operation.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.graph_features[idx]),
        )


class PredictorToxiPep:
    """
    ToxiPep predictor for peptide toxicity assessment

    This predictor uses a hybrid Transformer-CNN architecture to predict whether
    peptides are toxic or non-toxic. It combines sequence embeddings with molecular
    graph features for improved prediction accuracy.

    The model outputs toxicity probabilities where:
    - Higher values (closer to 1.0) indicate higher toxicity
    - Lower values (closer to 0.0) indicate lower toxicity
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        model: str = "default",
        models_directory: str | Path | None = None,
    ) -> None:
        """Initialize the ToxiPep predictor

        Args:
            device (str): Device for computation ("cpu" or "cuda")
            model (str): Registered ToxiPep model variant.
            models_directory (str, optional): Replacement strategy package root.

        :param device: Computation device accepted by the numerical backend.
        :type device: str | torch.device
        :param model: Named model variant.
        :type model: str
        :param models_directory: Optional external model root replacing packaged artifacts.
        :type models_directory: str | Path | None
        :rtype: None
        """
        self.device = torch.device(device)

        # Model configuration - same as in original ToxiPep
        self.vocab_size = len(Pep_residue2idx)
        self.d_model = 256
        self.d_ff = 512
        self.n_layers = 2
        self.n_heads = 4
        self.max_len = 50

        self.structural_config = {
            "embedding_dim": 21,
            "max_seq_len": 50,
            "filter_num": 64,
            "filter_sizes": [(3, 3), (5, 5), (7, 7), (9, 9)],
        }

        # Initialize model
        self.model = ToxiPep_Model(
            self.vocab_size,
            self.d_model,
            self.d_ff,
            self.n_layers,
            self.n_heads,
            self.max_len,
            structural_config=self.structural_config,
        ).to(self.device)

        # Model resolution and loading
        model_paths = ARTIFACTS.resolve(
            model,
            root=models_directory,
        )
        model_path = model_paths[0]
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"ToxiPep model loaded from: {model_path}")

    def transform_sequences_to_index(self, sequences: Sequence[str]) -> list[object]:
        """Convert peptide sequences to index representation

        :param sequences: Peptide sequences in input order.
        :type sequences: Sequence[str]
        :return: Result of this operation.
        :rtype: list[object]
        """
        token_index = []
        for seq in sequences:
            seq_id = [Pep_residue2idx.get(residue, 0) for residue in seq]
            token_index.append(seq_id)
        return token_index

    def pad_sequences(
        self, token_list: Sequence[Sequence[int]], max_len: int = 51
    ) -> list[list[int]]:
        """Pad sequences to fixed length with [CLS] token

        :param token_list: Token list.
        :type token_list: Sequence[Sequence[int]]
        :param max_len: Max len.
        :type max_len: int
        :return: Result of this operation.
        :rtype: list[list[int]]
        """
        data = []
        for seq in token_list:
            seq = [Pep_residue2idx["[CLS]"]] + seq
            seq.extend([Pep_residue2idx["[PAD]"]] * (max_len - len(seq)))
            data.append(seq[:max_len])
        return data

    def preprocess_sequences(
        self, sequences: Sequence[str]
    ) -> tuple[object, list[object]]:
        """Preprocess peptide sequences for prediction

        Args:
            sequences (list): List of peptide sequences as strings

        Returns:
            tuple: (padded_sequences, graph_features)

        :param sequences: Peptide sequences in input order.
        :type sequences: Sequence[str]
        :return: Result of this operation.
        :rtype: tuple[object, list[object]]
        """
        # Convert to index representation
        indexed_sequences = self.transform_sequences_to_index(sequences)

        # Pad sequences
        padded_sequences = self.pad_sequences(indexed_sequences)

        # Generate graph features
        graph_features = [convert_to_graph_channel(seq) for seq in sequences]

        return padded_sequences, graph_features

    def predict_batch(
        self, sequences: Sequence[str], batch_size: int = 64
    ) -> np.ndarray:
        """Predict toxicity for a batch of sequences

        Args:
            sequences (list): List of peptide sequences as strings
            batch_size (int): Batch size for processing

        Returns:
            tuple: (predictions, probabilities)
                - predictions: Binary predictions (0=non-toxic, 1=toxic)
                - probabilities: Toxicity probabilities (0-1 range)

        :param sequences: Peptide sequences in input order.
        :type sequences: Sequence[str]
        :param batch_size: Batch size.
        :type batch_size: int
        :return: Result of this operation.
        :rtype: np.ndarray
        """
        # Preprocess sequences
        padded_sequences, graph_features = self.preprocess_sequences(sequences)

        # Create dataset and dataloader
        dataset = PeptideDataset(padded_sequences, graph_features)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Make predictions
        self.model.eval()
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for input_ids, graph_features_batch in dataloader:
                input_ids = input_ids.to(self.device)
                graph_features_batch = graph_features_batch.to(self.device)

                # Forward pass
                outputs = self.model(input_ids, graph_features_batch, self.device)

                # Get probabilities and predictions
                probs = torch.softmax(outputs, dim=1)[
                    :, 1
                ]  # Positive class probability
                preds = outputs.argmax(dim=1)

                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())

        return np.array(all_predictions), np.array(all_probabilities)

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        """Predict toxicity for input sequences (main interface method)

        Args:
            sequences (list): List of peptide sequences as strings

        Returns:
            np.ndarray: Toxicity probabilities with shape (n_sequences, 1)
                       Higher values indicate higher toxicity probability

        :param sequences: Peptide sequences in input order.
        :type sequences: Sequence[str]
        :return: Result of this operation.
        :rtype: np.ndarray
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        predictions, probabilities = self.predict_batch(sequences)

        # Return probabilities in the same format as other predictors
        return probabilities.reshape(-1, 1)

    def get_predictions_with_details(self, sequences: Sequence[str]) -> pd.DataFrame:
        """Get detailed prediction results including binary classifications

        Args:
            sequences (list): List of peptide sequences as strings

        Returns:
            pd.DataFrame: DataFrame with columns:
                - sequence: Input peptide sequence
                - toxicity_prob: Probability of being toxic (0-1)
                - prediction: Binary classification (0=non-toxic, 1=toxic)
                - toxicity_label: Human-readable label

        :param sequences: Peptide sequences in input order.
        :type sequences: Sequence[str]
        :return: Result of this operation.
        :rtype: pd.DataFrame
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        predictions, probabilities = self.predict_batch(sequences)

        results = pd.DataFrame(
            {
                "sequence": sequences,
                "toxicity_prob": probabilities,
                "prediction": predictions,
                "toxicity_label": [
                    "Toxic" if pred == 1 else "Non-toxic" for pred in predictions
                ],
            }
        )

        return results

    def __call__(self, sequences: Sequence[str]) -> np.ndarray:
        """Make the predictor callable (same as predict method)

        :param sequences: Peptide sequences in input order.
        :type sequences: Sequence[str]
        :return: Result of this operation.
        :rtype: np.ndarray
        """
        return self.predict(sequences)


# Example usage and testing
if __name__ == "__main__":
    print("Testing ToxiPepPredictor...")

    # Initialize predictor
    predictor = PredictorToxiPep(device="cpu")

    # Test sequences
    test_sequences = [
        "KLLLKLLKKLLKLLK",
        "FLPIIAKLLGLL",
        "WLGHFTVRK",
        "ALWKTLLKKVLKAPKLLK",
        "GGGGGGGGGGG",
    ]

    # Test main predict method
    print("\nTesting predict() method:")
    predictions = predictor.predict(test_sequences)
    print(f"Predictions shape: {predictions.shape}")
    print("Toxicity probabilities:")
    for i, (seq, prob) in enumerate(zip(test_sequences, predictions)):
        print(f"  {seq}: {prob[0]:.4f}")

    # Test detailed predictions
    print("\nTesting get_predictions_with_details() method:")
    detailed_results = predictor.get_predictions_with_details(test_sequences)
    print(detailed_results.to_string(index=False))

    # Test single sequence
    print("\nTesting single sequence prediction:")
    single_pred = predictor("KLLLKLLKKLLKLLK")
    print(f"Single prediction: {single_pred[0][0]:.4f}")

    print("\n✓ ToxiPepPredictor test completed successfully!")
