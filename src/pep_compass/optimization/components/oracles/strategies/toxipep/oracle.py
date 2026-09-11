"""
ToxiPep Black Box for POLI Optimization Framework

This module provides a black box interface for the ToxiPep toxicity prediction model,
allowing it to be used with optimization frameworks like POLI for peptide design tasks.
"""

import numpy as np
from typing import Optional

try:
    from poli_baselines.core.abstract_solver import AbstractBlackBox
    from poli.core.black_box_information import BlackBoxInformation
except ImportError:
    # Fallback for when POLI is not available
    class AbstractBlackBox:
        def __init__(self, **kwargs):
            pass
    BlackBoxInformation = dict

from pep_compass.optimization.components.oracles.strategies.toxipep.ToxiPepPredictor import PredictorToxiPep

class ToxiPepBlackBox(AbstractBlackBox):
    """
    TODO: review docstring. It is inconssistent with the PredictorToxiPep docstring.
    
    Black box interface for ToxiPep toxicity prediction model
    
    This class provides a standardized interface for the ToxiPep model within
    optimization frameworks like POLI. It predicts peptide toxicity and returns
    negative toxicity probabilities as scores, so optimization algorithms
    seeking to maximize scores will find peptides with lower toxicity.
    """
    
    def __init__(
        self,
        device: str = "cpu",
        batch_size: Optional[int] = None,
        parallelize: bool = False,
        num_workers: Optional[int] = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
        model: str = "default",
        models_directory: str | None = None,
    ):
        """
        Initialize the ToxiPep Black Box
        
        Args:
            device (str): Device for computation ("cpu" or "cuda")
            batch_size (int, optional): Batch size for processing sequences
            parallelize (bool): Enable parallel processing
            num_workers (int, optional): Number of worker processes
            evaluation_budget (int): Maximum number of evaluations allowed
            force_isolation (bool): Force process isolation
        """
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        
        # Initialize the ToxiPep predictor
        self.toxipep_predictor = PredictorToxiPep(
            device=device,
            model=model,
            models_directory=models_directory,
        )
        
        # Define the peptide scoring function
        # Returns NEGATIVE toxicity probabilities for minimization (lower toxicity = higher score)
        self.peptide_scorer = lambda sequences: self.toxipep_predictor.predict(sequences).flatten()
        
        # Cache for storing results
        self.cache = []
        
        self.maximize = False 
        
        print(f"ToxiPep Black Box initialized on {device}")
    
    def get_black_box_info(self) -> BlackBoxInformation:
        """Return information about this black box for POLI framework"""
        return BlackBoxInformation(
            name="ToxiPep",
            max_sequence_length=50,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=list("ACDEFGHIKLMNPQRSTVWY"),
            log_transform_recommended=False,
            discrete=True,
            padding_token=" ",
        )
    
    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        """
        Evaluate peptide sequences for toxicity (main evaluation method)
        
        Args:
            x (np.ndarray): Array of peptide sequences as character arrays
                          Shape: (n_sequences, sequence_length)
            context (dict, optional): Additional context information
                          
        Returns:
            np.ndarray: Toxicity scores with shape (n_sequences, 1)
                       Higher scores indicate lower toxicity (safer peptides)
        """
        # Convert character arrays to sequence strings
        sequences = ["".join(seq).strip() for seq in x]
        
        # Remove any non-alphabetic characters (padding, etc.)
        sequences = [''.join(c for c in seq if c.isalpha()) for seq in sequences]
        
        # Get toxicity predictions and convert to scores
        
        predictions = self.peptide_scorer(sequences)
        
        # Cache results
        for i, seq in enumerate(sequences):
            self.cache.append((seq, predictions[i].item()))
        
        return predictions.reshape(-1, 1)
    
    def clear_cache(self):
        """Clear the results cache"""
        self.cache = []
    
    def get_cache_size(self) -> int:
        """Get the current cache size"""
        return len(self.cache)
    
    def get_cached_results(self) -> list:
        """Get all cached results as (sequence, score) tuples"""
        return self.cache.copy()

# Example usage and testing
if __name__ == "__main__":
    print("Testing ToxiPep Black Box...")
    
    # Initialize black box
    black_box = ToxiPepBlackBox(device="cpu")
    
    # Test sequences as character arrays (POLI format)
    test_sequences = [
        "KLLLKLLKKLLKLLK",
        "FLPIIAKLLGLL",
        "WLGHFTVRK",
        "ALWKTLLKKVLKAPKLLK",
        "GGGGGGGGGGG"
    ]
    
    # Convert to character arrays and pad to same length
    max_len = max(len(seq) for seq in test_sequences)
    padded_sequences = []
    for seq in test_sequences:
        char_array = list(seq) + [' '] * (max_len - len(seq))
        padded_sequences.append(char_array)
    
    x = np.array(padded_sequences)
    
    print(f"Input shape: {x.shape}")
    print("Test sequences:")
    for i, seq in enumerate(test_sequences):
        print(f"  {i+1}. {seq}")
    
    # Get predictions
    try:
        scores = black_box._black_box(x)
        print(f"\nScores shape: {scores.shape}")
        print("Results (higher scores = higher toxicity probability):")
        for i, (seq, score) in enumerate(zip(test_sequences, scores)):
            print(f"  {seq:<20}: {score[0]:.4f}")
        
        print(f"\nCache size: {black_box.get_cache_size()}")
        
        # Test get_black_box_info
        info = black_box.get_black_box_info()
        print(f"\nBlack box info: {info}")
        
        print("\n✓ ToxiPep Black Box test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
