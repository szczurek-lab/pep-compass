from __future__ import annotations

import numpy as np
import torch
from poli.core.black_box_information import BlackBoxInformation
from poli_baselines.core.abstract_solver import AbstractBlackBox

from pep_compass.steps.oracles.strategies.hydrophobicity.HydrophobicityPredictor import (
    HydrophobicityPredictor,
)


class HydrophobicityBlackBox(AbstractBlackBox):
    def __init__(
        self,
        *,
        batch_size: int | None = None,
        parallelize: bool = False,
        num_workers: int | None = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
        scale: str = "eisenberg",
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
        :param scale: Scale.
        :type scale: str
        :rtype: None
        """
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )

        self.hydrophobicity_predictor = HydrophobicityPredictor(scale=scale)

        # Create scoring function - higher hydrophobicity is better for optimization
        self.peptide_scorer = lambda x: self.hydrophobicity_predictor.predict(
            x
        ).flatten()

        self.cache = []
        self.maximize = True  # Higher hydrophobicity is better

    def get_black_box_info(self) -> BlackBoxInformation:
        """Get black box info.

        :return: Result of this operation.
        :rtype: BlackBoxInformation
        """
        return BlackBoxInformation(
            name="Hydrophobicity",
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


if __name__ == "__main__":
    # Test the hydrophobicity black box
    print("Testing Hydrophobicity Black Box...")

    try:
        # Initialize black box
        bb = HydrophobicityBlackBox(scale="eisenberg")
        print("Hydrophobicity Black Box initialized")

        # Test sequences as arrays (like the black box expects)
        test_sequences = [
            list("KLLLKLLKKLLKLLK"),
            list("FLPIIAKLLGLL"),
            list("WLGHFTVRK"),
            list("RRRRRRR"),
            list("LLLLLLL"),
        ]
        test_array = np.array(test_sequences, dtype=object)

        print(f"Input shape: {test_array.shape}")
        print("Test sequences:")
        for i, seq in enumerate(test_sequences):
            seq_str = "".join(seq)
            print(f"  {i + 1}. {seq_str}")

        # Run predictions
        results = bb._black_box(test_array)

        print(f"\nScores shape: {results.shape}")
        print("Results (higher scores = more hydrophobic):")
        for i, (seq, score) in enumerate(zip(test_sequences, results.flatten())):
            seq_str = "".join(seq)
            print(f"  {seq_str:15s}: {score:7.4f}")

        print(f"\nCache size: {len(bb.cache)}")

        # Print black box info
        info = bb.get_black_box_info()
        print(f"\nBlack box info: {info}")

        print("\n✓ Hydrophobicity Black Box test completed successfully!")

    except Exception as e:
        print(f"✗ Error testing Hydrophobicity Black Box: {e}")
        import traceback

        traceback.print_exc()
