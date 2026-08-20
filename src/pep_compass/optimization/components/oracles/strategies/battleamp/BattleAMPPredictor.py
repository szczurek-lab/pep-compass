import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
from tqdm import tqdm
from .utils import chopping, padding, onehot_encoding, prepare_CNN
from pep_compass.optimization.components.oracles.model_registry import (
    resolve_oracle_model,
)


class PredictorBattleAMP:
    """
    BattleAMP predictor following the same interface as APEX predictor.
    Predicts antimicrobial activity (MIC values) for peptide sequences.
    """

    def __init__(
        self,
        device="cpu",
        batch_size=3000,
        model="default",
        models_directory=None,
    ):
        """
        Initialize the BattleAMP predictor.
        
        Args:
            device (str): Device to use for computation ("cpu" or "cuda"). 
                         Note: TensorFlow will handle GPU allocation automatically.
            batch_size (int): Batch size for processing sequences.
            model (str): Registered BattleAMP model variant.
            models_directory (str, optional): Replacement strategy package root.
        """
        self.device = device
        self.batch_size = batch_size
        self.model_name = model

        # Model resolution
        ## Validate the complete named artifact before TensorFlow initialization
        _, model_paths = resolve_oracle_model(
            "battleamp",
            model,
            models_directory=models_directory,
        )
        model_path = model_paths[0]

        if str(device).startswith("cpu"):
           # Max - debbuging: Zmiana z pozostawienia TensorFlow dostępu do wykrytych GPU na jawne ukrycie GPU ~BattleAMP działa w osobnym procesie na CPU, więc TensorFlow nie może inicjalizować CUDA ani rezerwować pamięci GPU używanej przez PyTorch.
            tf.config.set_visible_devices([], "GPU")
            
        # Configure TensorFlow to use GPU if available and requested
        if device == "cuda":
            gpus = tf.config.experimental.list_physical_devices('GPU')
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
        self.pathogen_list = ["Gram -"]

    def predict(self, seq_list, use_tqdm: bool = False):
        """
        Predict antimicrobial activity (MIC values) for a list of peptide sequences.
        
        Args:
            seq_list (list): List of peptide sequences as strings.
            use_tqdm (bool): Whether to show progress bars.
            
        Returns:
            np.ndarray: Predicted MIC values with shape (N, 1) where N is the number of sequences.
                       Values are in μM units.
        """
        data_len = len(seq_list)
        
        # Progress bar setup
        pbar = tqdm(total=data_len, desc="Processing sequences") if use_tqdm else None
        
        # Process sequences in batches
        batch_iter = range(int(math.ceil(data_len / float(self.batch_size))))
        all_predictions = []
        
        for i in batch_iter:
            
            seq_batch = seq_list[i * self.batch_size : (i + 1) * self.batch_size]
            batch_input = prepare_CNN(seq_batch) 
            
            
            batch_predictions = self.model(batch_input).numpy()  
            all_predictions.append(batch_predictions)
            
         
            if pbar is not None:
                pbar.update(len(seq_batch))
        
        if pbar is not None:
            pbar.close()
            
     
        predictions = np.vstack(all_predictions)
        
        # Convert from log10 MIC to normal MIC values
        predictions = 10 ** predictions
        
        return predictions
