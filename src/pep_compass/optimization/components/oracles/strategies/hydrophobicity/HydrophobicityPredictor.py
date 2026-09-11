import numpy as np
from typing import List, Union
from modlamp.descriptors import PeptideDescriptor


class HydrophobicityPredictor:
    """
    Hydrophobicity predictor using a modlAMP amino-acid scale.
    Compatible interface with other predictors (APEX, BattleAMP, etc.).
    """
    
    def __init__(self, scale: str = "eisenberg"):
        """
        Initialize the Hydrophobicity predictor using modlAMP.
        
        Args:
            scale (str): Hydrophobicity scale to use. Supports scales available in modlAMP.
        """
        self.scale = scale
        
        # Set pathogen list to match other predictors
        self.pathogen_list = ["Hydrophobicity"]

    def predict(self, seq_list: List[str], use_tqdm: bool = False) -> np.ndarray:
        """
        Predict hydrophobicity scores for a list of peptide sequences using modlAMP.
        
        Args:
            seq_list (list): List of peptide sequences as strings.
            use_tqdm (bool): Whether to show progress bars (not used in this implementation).
            
        Returns:
            np.ndarray: Predicted hydrophobicity scores with shape (N, 1) where N is the number of sequences.
                       Higher values indicate more hydrophobic peptides.
        """
        scores = []
        
        for seq in seq_list:
            try:
                # Max - debbuging: Zmiana z wywołania modelu hydrophobicity przez seqme na bezpośrednie użycie modlamp.PeptideDescriptor ~seqme korzystało z tego samego backendu, ale podczas importu ładowało również TensorFlow; teraz skala i sposób obliczania wyniku pozostają takie same, bez niepotrzebnej inicjalizacji TensorFlow i konfliktu z CUDA PyTorch.
                # Link do dokumentacji: `https://seqme.readthedocs.io/en/stable/_modules/seqme/models/amino_acid_descriptors.html#Hydrophobicity`
                descriptor = PeptideDescriptor(seq)
                descriptor.load_scale(self.scale)
                descriptor.calculate_global()
                score = descriptor.descriptor.squeeze(axis=-1)
                scores.append(score)
            except Exception as e:
                print(f"Warning: Error scoring sequence '{seq}': {e}. Using -100")
                scores.append(-100.0)
        
        # Return as numpy array with shape (N, 1) to match other predictors
        return np.array(scores).reshape(-1, 1)

    def get_available_scales(self) -> List[str]:
        """
        Get list of available hydrophobicity scales.
        
        Returns:
            List[str]: Available scales exposed by the predictor.
        """
        try:
            return PeptideDescriptor.scalenames
        except:
            return ["eisenberg"]  # fallback
    
    def get_scale_info(self) -> dict:
        """
        Get information about the hydrophobicity scale being used.
        
        Returns:
            dict: Information about the scale including name and available scales.
        """
        return {
            "current_scale": self.scale,
            "available_scales": self.get_available_scales(),
            "description": f"Hydrophobicity predictor using modlAMP with {self.scale} scale",
            # Max - debbuging: Zmiana z usuniętego atrybutu seqme na aktywną klasę deskryptora ~get_scale_info kończyło się AttributeError.
            "seqme_model": PeptideDescriptor.__name__,
        }


if __name__ == "__main__":
    # Test the hydrophobicity predictor with modlAMP
    print("=== modlAMP Hydrophobicity Predictor Test ===")
    
    try:
        predictor = HydrophobicityPredictor(scale="eisenberg")
        
        # Test sequences
        test_sequences = [
            "KLLLKLLKKLLKLLK",  # More hydrophobic (lots of L)
            "FLPIIAKLLGLL",     # Mixed hydrophobicity
            "WLGHFTVRK",        # Mixed
            "RRRRRRR",          # Very hydrophilic (all R)
            "LLLLLLL",          # Very hydrophobic (all L)
            "GGGGGGG"           # Neutral (all G)
        ]
        
        print(f"Scale info: {predictor.get_scale_info()}")
        print(f"Pathogen list: {predictor.pathogen_list}")
        
        print("\nTesting predictions:")
        predictions = predictor.predict(test_sequences)
        
        for i, (seq, score) in enumerate(zip(test_sequences, predictions.flatten())):
            print(f"{i+1:2d}. {seq:15s} -> Hydrophobicity: {score:6.3f}")
        
        print(f"\nPredictions shape: {predictions.shape}")
        print("✓ seqme Hydrophobicity predictor test completed successfully!")
        
    except Exception as e:
        print(f"Error testing hydrophobicity predictor: {e}")
        import traceback
        traceback.print_exc()
