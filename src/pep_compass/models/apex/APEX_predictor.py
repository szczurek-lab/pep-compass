import glob
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pep_compass.models.apex.APEX_models import AMP_model
from pep_compass.models.apex.utils import make_vocab, onehot_encoding


class APEXUnpickler(pickle.Unpickler):
    """Custom unpickler that maps old module names to current ones."""
    
    def find_class(self, module, name):
        # Map old top-level module names to the correct package paths
        if module == "APEX_models":
            module = "pep_compass.models.apex.APEX_models"
        elif module == "AMP_DL_model_twohead":
            module = "pep_compass.models.apex.APEX_models"
        return super().find_class(module, name)


# Create a simple module-like object for torch.load
class APEXPickleModule:
    """Custom pickle module for torch.load."""
    Unpickler = APEXUnpickler
    # Delegate everything else to pickle
    def __getattr__(self, name):
        return getattr(pickle, name)


class PredictorAPEX:

    def __init__(self, device="cpu", batch_size=3000, path="default", models_dir: str | None = None):
        self.device = device
        self.path = path
        self.models_dir = models_dir
        if path == "default":
            self.pathogen_list = [
                "A. baumannii ATCC 19606",
                "E. coli ATCC 11775",
                "E. coli AIG221",
                "E. coli AIG222",
                "K. pneumoniae ATCC 13883",
                "P. aeruginosa PA01",
                "P. aeruginosa PA14",
                "S. aureus ATCC 12600",
                "S. aureus (ATCC BAA-1556) - MRSA",
                "vancomycin-resistant E. faecalis ATCC 700802",
                "vancomycin-resistant E. faecium ATCC 700221",
            ]
        elif path == "all":
            self.pathogen_list = [
                "A. baumannii ATCC 19606",
                "E. coli ATCC 11775",
                "E. coli AIG221",
                "E. coli AIG222",
                "K. pneumoniae ATCC 13883",
                "P. aeruginosa PA01",
                "P. aeruginosa PA14",
                "S. aureus ATCC 12600",
                "S. aureus (ATCC BAA-1556) - MRSA",
                "vancomycin-resistant E. faecalis ATCC 700802",
                "vancomycin-resistant E. faecium ATCC 700221",
                # Additional bacteria from APEX_FULL
                "A. muciniphila ATCC BAA-835",
                "B. fragilis ATCC25285",
                "B. vulgatus ATCC8482",
                "C. aerofaciens ATCC25986",
                "C. scindens ATCC35704",
                "B. thetaiotaomicron ATCC29148",
                "B. thetaiotaomicron Complemmented",
                "B. thetaiotaomicron Mutant",
                "B. uniformis ATCC8492",
                "B. eggerthi ATCC27754",
                "C. spiroforme ATCC29900",
                "P. distasonis ATCC8503",
                "P. copri DSMZ18205",
                "B. ovatus ATCC8483",
                "E. rectale ATCC33656",
                "C. symbiosum",
                "R. obeum",
                "R. torques",
                "E. coli Nissle",
                "Salmonella enterica ATCC 9150 (BEIRES NR-515)",
                "Salmonella enterica (BEIRES NR-170)",
                "Salmonella enterica ATCC 9150 (BEIRES NR-174)",
                "L. monocytogenes ATCC 19111 (BEIRES NR-106)",
            ]

        else:
            raise ValueError("Path option not recognized. Use 'default' or 'all'.")

        self.max_len = 52  # maximum seq length; 52 = start character + maximum peptide length (50 aa) + end character; longer peptides will be truncated
        self.word2idx, self.idx2word = make_vocab()  # make amino acid vocabulary
        # emb, AAindex_dict = AAindex('./aaindex1.csv', word2idx) #make amino acid embeddings

        # Load pretrained APEX models (8 in total)
        # Use custom pickle module to handle old module name references
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.APEX_models = []
        if path == "default":
            if self.models_dir is not None:
                self.file_dir = self.models_dir
            if not Path(f"{self.file_dir}/APEX_pathogen_models").exists():
                raise FileNotFoundError(
                    f"Directory {self.file_dir}/APEX_pathogen_models with APEX pathogen models not found. "
                   "Please copy the folder from https://github.com/Yimeng-Zeng/APEXGo/tree/main/optimization/apex_oracle/APEX_pathogen_models"
                )
            
            for a_model in glob.glob(f"{self.file_dir}/APEX_pathogen_models/APEX_*"):
                model = torch.load(
                    a_model,
                    map_location=torch.device(self.device),
                    weights_only=False,
                    pickle_module=APEXPickleModule,
                )
                model.eval()
                self.APEX_models.append(model)
        elif path == "all":
            for a_model in glob.glob(f"{self.file_dir}/Full_APEX_pathogen_models/trained_*"):
                model = torch.load(
                    a_model,
                    map_location=torch.device(self.device),
                    weights_only=False,
                    pickle_module=APEXPickleModule,
                )
                model.eval()
                self.APEX_models.append(model)

        self.batch_size = batch_size  # change according to your GPU memory

    # Use pretrained APEX models to predict species-specific antimicrobial activity (i.e., minimum inhibitory concentration [MIC]; unit: uM)
    # 8 pretrained APEX models are provided, and predictions are averaged
    def predict(self, seq_list, use_tqdm: bool = False):
        AMP_sum = 0

        data_len = len(seq_list)
        num_models = len(self.APEX_models)
        outer_bar = tqdm(total=num_models, desc="Models") if use_tqdm else None

        for ensemble_id in range(num_models):
            AMP_model = self.APEX_models[ensemble_id].to(self.device).eval()

            inner_bar = (
                tqdm(
                    total=data_len,
                    desc=f"Sequences [{ensemble_id + 1}/{num_models}]",
                    leave=False,
                )
                if use_tqdm
                else None
            )

            batch_iter = range(int(math.ceil(data_len / float(self.batch_size))))
            for i in batch_iter:
                seq_batch = seq_list[i * self.batch_size : (i + 1) * self.batch_size]
                seq_rep = onehot_encoding(
                    seq_batch, self.max_len, self.word2idx
                )  # make input
                X_seq = torch.LongTensor(seq_rep).to(self.device)

                AMP_pred_batch = (
                    AMP_model(X_seq).cpu().detach().numpy()
                )  # make predictions
                AMP_pred_batch = 10 ** (
                    6 - AMP_pred_batch
                )  # transform back to MICs; When training the APEX models, MICs were transformed by: -np.log10(MICs/float(1000000))

                if i == 0:
                    AMP_pred = AMP_pred_batch
                else:
                    AMP_pred = np.vstack([AMP_pred, AMP_pred_batch])

                if inner_bar is not None:
                    inner_bar.update(len(seq_batch))

            # sum up the predictions made by different APEX models
            if ensemble_id == 0:
                AMP_sum = AMP_pred
            else:
                AMP_sum += AMP_pred

            if inner_bar is not None:
                inner_bar.close()
            if outer_bar is not None:
                outer_bar.update(1)

        if outer_bar is not None:
            outer_bar.close()

        AMP_pred = AMP_sum / float(len(self.APEX_models))  # average the predictions

        return AMP_pred


class ExpandSequenceFunction(torch.autograd.Function):
    """Custom autograd function for expanding sequences with gradient support."""

    @staticmethod
    def forward(ctx, X_seq):
        # Save original shape for backward pass
        original_shape = X_seq.shape
        ctx.original_shape = original_shape

        # Reshape input to expected format (25, 21)
        X_seq = X_seq.view(25, 21)

        # Save reshaped input for backward pass
        ctx.save_for_backward(X_seq)

        seq_len, vocab_size = X_seq.shape  # (25, 21)
        new_vocab_size = vocab_size + 2  # Expanding from 21 → 23
        new_seq_len = 52

        expanded_probs = torch.zeros(
            (new_seq_len, new_vocab_size), dtype=X_seq.dtype, device=X_seq.device
        )
        argmax_indices = torch.argmax(X_seq, dim=1)
        if torch.sum(argmax_indices == 0) == 0:
            end_ind = seq_len
        else:
            non_zero_indices = (argmax_indices == 0).nonzero(as_tuple=True)[0]
            end_ind = torch.min(non_zero_indices).item()  # Convert to Python scalar

        # we want to correctly change the probles to be corresponding to their one hot encoding
        expanded_probs[1 : (end_ind + 1), 0] = X_seq[:end_ind, 0]  # probs for padding
        expanded_probs[1 : (end_ind + 1), 3:] = X_seq[:end_ind, 1:]  # rest of our probs

        # starting padding
        expanded_probs[0, 1] = 1  # First row gets start padding

        # we assume rest is padding
        if end_ind + 2 < new_seq_len:
            expanded_probs[end_ind + 2 :, 0] = 1

        # last_non_zero_index+1 - we change it index to have prob of end padding = 1
        expanded_probs[end_ind + 1, 2] = 1

        # Store end_ind and dimensions for backward pass
        ctx.end_ind = end_ind
        ctx.seq_len = seq_len
        ctx.vocab_size = vocab_size

        return expanded_probs

    @staticmethod
    def backward(ctx, grad_output):
        (X_seq,) = ctx.saved_tensors
        end_ind = ctx.end_ind
        original_shape = ctx.original_shape

        # Initialize gradient for input with zeros using the correct shape
        grad_X_seq = torch.zeros_like(X_seq)

        # Safely handle the gradient backpropagation
        end_idx = min(end_ind, X_seq.size(0))

        # Backpropagate gradients for padding probabilities (index 0)
        if end_idx > 0:
            grad_X_seq[:end_idx, 0] = grad_output[1 : end_idx + 1, 0]

        # Backpropagate gradients for content (indices 1 onwards -> 3 onwards in expanded)
        if end_idx > 0:
            # Make sure dimensions align
            content_grad = grad_output[1 : end_idx + 1, 3 : 3 + X_seq.size(1) - 1]
            if content_grad.size(1) == X_seq.size(1) - 1:
                grad_X_seq[:end_idx, 1:] = content_grad

        # Return gradient reshaped to match the original input shape
        return grad_X_seq.view(original_shape)


class PredictorAPEX_Probs:
    """
    Input as probs, not sequence
    """

    def __init__(self, device="cpu", batch_size=3000, path="default"):
        self.device = device
        self.path = path
        if path == "default":
            self.pathogen_list = [
                "A. baumannii ATCC 19606",
                "E. coli ATCC 11775",
                "E. coli AIG221",
                "E. coli AIG222",
                "K. pneumoniae ATCC 13883",
                "P. aeruginosa PA01",
                "P. aeruginosa PA14",
                "S. aureus ATCC 12600",
                "S. aureus (ATCC BAA-1556) - MRSA",
                "vancomycin-resistant E. faecalis ATCC 700802",
                "vancomycin-resistant E. faecium ATCC 700221",
            ]

        elif path == "all":
            self.pathogen_list = [
                "A. baumannii ATCC 19606",
                "E. coli ATCC 11775",
                "E. coli AIG221",
                "E. coli AIG222",
                "K. pneumoniae ATCC 13883",
                "P. aeruginosa PA01",
                "P. aeruginosa PA14",
                "S. aureus ATCC 12600",
                "S. aureus (ATCC BAA-1556) - MRSA",
                "vancomycin-resistant E. faecalis ATCC 700802",
                "vancomycin-resistant E. faecium ATCC 700221",
                # Additional bacteria from APEX_FULL
                "A. muciniphila ATCC BAA-835",
                "B. fragilis ATCC25285",
                "B. vulgatus ATCC8482",
                "C. aerofaciens ATCC25986",
                "C. scindens ATCC35704",
                "B. thetaiotaomicron ATCC29148",
                "B. thetaiotaomicron Complemmented",
                "B. thetaiotaomicron Mutant",
                "B. uniformis ATCC8492",
                "B. eggerthi ATCC27754",
                "C. spiroforme ATCC29900",
                "P. distasonis ATCC8503",
                "P. copri DSMZ18205",
                "B. ovatus ATCC8483",
                "E. rectale ATCC33656",
                "C. symbiosum",
                "R. obeum",
                "R. torques",
                "E. coli Nissle",
                "Salmonella enterica ATCC 9150 (BEIRES NR-515)",
                "Salmonella enterica (BEIRES NR-170)",
                "Salmonella enterica ATCC 9150 (BEIRES NR-174)",
                "L. monocytogenes ATCC 19111 (BEIRES NR-106)",
            ]

        else:
            raise ValueError("Path option not recognized. Use 'default' or 'all'.")

        self.max_len = 52  # maximum seq length; 52 = start character + maximum peptide length (50 aa) + end character; longer peptides will be truncated
        self.word2idx, self.idx2word = make_vocab()  # make amino acid vocabulary
        # emb, AAindex_dict = AAindex('./aaindex1.csv', word2idx) #make amino acid embeddings

        # Load pretrained APEX models (8 in total for default, 40 for full)
        # Use custom pickle module to handle old module name references
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.APEX_models = []
        if path == "default":
            for a_model in glob.glob(f"{self.file_dir}/APEX_pathogen_models/APEX_*"):
                model = torch.load(
                    a_model,
                    map_location=torch.device(self.device),
                    weights_only=False,
                    pickle_module=APEXPickleModule,
                )
                self.APEX_models.append(model)
        elif path == "all":
            for a_model in glob.glob(f"{self.file_dir}/Full_APEX_pathogen_models/trained_*"):
                model = torch.load(
                    a_model,
                    map_location=torch.device(self.device),
                    weights_only=False,
                    pickle_module=APEXPickleModule,
                )
                self.APEX_models.append(model)
        self.batch_size = batch_size  # change according to your GPU memory

    # Use pretrained APEX models to predict species-specific antimicrobial activity (i.e., minimum inhibitory concentration [MIC]; unit: uM)
    # 8 pretrained APEX models or 40 for Full APEX are provided, and predictions are averaged
    def predict(self, probs, return_x=False, use_tqdm: bool = False):
        AMP_sum = 0
        expanded_probs_saved = None

        data_len = probs.shape[0]
        pbar = tqdm(total=data_len, desc="Sequences") if use_tqdm else None

        for ensemble_id in range(len(self.APEX_models)):
            # Train mode is necessary to allow gradient computation
            AMP_model = self.APEX_models[ensemble_id].to(self.device).train()

            batch_iter = range(int(math.ceil(data_len / float(self.batch_size))))
            for i in batch_iter:
                seq_batch = probs[i * self.batch_size : (i + 1) * self.batch_size]
                X_seq = seq_batch.to(self.device)

                # Apply the custom autograd function to expand the sequence
                expanded_probs = self.expand_sequence(X_seq)

                # Save expanded_probs for the first batch if return_x is True
                if return_x and ensemble_id == 0 and i == 0:
                    expanded_probs_saved = expanded_probs.clone()

                # Reshape to match the expected input shape
                expanded_probs = expanded_probs.view(-1, 52, 23)

                AMP_pred_batch = AMP_model.forward_for_probs(
                    expanded_probs
                )  # make predictions
                AMP_pred_batch = 10 ** (
                    6 - AMP_pred_batch
                )  # transform back to MICs; When training the APEX models, MICs were transformed by: -np.log10(MICs/float(1000000))

                if i == 0:
                    AMP_pred = AMP_pred_batch
                else:
                    AMP_pred = torch.vstack([AMP_pred, AMP_pred_batch])

            AMP_pred = torch.cat(AMP_pred, dim=0)

            if AMP_sum is None:
                AMP_sum = AMP_pred
            else:
                AMP_sum = AMP_sum + AMP_pred

            # progress update per batch for first ensemble only
            if pbar is not None and ensemble_id == 0:
                pbar.update(min(self.batch_size, data_len - i * self.batch_size))

        if pbar is not None:
            pbar.close()

        AMP_pred = AMP_sum / len(self.APEX_models)

        if return_x:
            return AMP_pred, expanded_probs_saved

        return AMP_pred

    def expand_sequence(self, X_seq):
        """
        Wrapper method to apply the custom autograd function for sequence expansion.
        This preserves gradient flow through the operation.
        """
        return ExpandSequenceFunction.apply(X_seq)
