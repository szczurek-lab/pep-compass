import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from pep_compass.optimization.components.oracles.strategies.apex_original.utils import make_vocab, onehot_encoding
from pep_compass.optimization.components.oracles.model_registry import (
    resolve_oracle_model,
)
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


_DEFAULT_PATHOGEN_LIST = [
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

_FULL_PATHOGEN_LIST = _DEFAULT_PATHOGEN_LIST + [
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


@dataclass(frozen=True)
class APEXModelVariant:
    """Describe one named, downloadable set of pretrained APEX weights.

    :param directory: Subdirectory of ``apex/models/`` holding the weight files.
    :param glob_pattern: Filename glob selecting weight files within that directory.
    :param pathogen_list: Pathogen names, aligned with each model's prediction columns.
    :param download_script: Repository-relative script that installs this variant.
    """

    directory: str
    glob_pattern: str
    pathogen_list: list[str] = field(default_factory=list)
    download_script: str = ""
    expected_models: int | None = None


APEX_MODEL_VARIANTS: dict[str, APEXModelVariant] = {
    "default": APEXModelVariant(
        directory="default",
        glob_pattern="APEX_*",
        pathogen_list=_DEFAULT_PATHOGEN_LIST,
        download_script="assets/scripts/downloads/apex/download_apex_models_default.sh",
        expected_models=8,
    ),
    "full": APEXModelVariant(
        directory="full",
        glob_pattern="trained_*",
        pathogen_list=_FULL_PATHOGEN_LIST,
        download_script="assets/scripts/downloads/apex/download_apex_models_full.sh",
        expected_models=40,
    ),
}


class APEXUnpickler(pickle.Unpickler):
    """Custom unpickler that maps old module names to current ones."""
    
    def find_class(self, module, name):
        # Map old top-level module names to the correct package paths
        if module == "APEX_models":
            module = "pep_compass.optimization.components.oracles.strategies.apex_original.APEX_models"
        elif module == "AMP_DL_model_twohead":
            module = "pep_compass.optimization.components.oracles.strategies.apex_original.APEX_models"
        return super().find_class(module, name)


# Create a simple module-like object for torch.load
class APEXPickleModule:
    """Custom pickle module for torch.load."""
    Unpickler = APEXUnpickler
    # Delegate everything else to pickle
    def __getattr__(self, name):
        return getattr(pickle, name)


class PredictorAPEX:
    """Load one APEX ensemble and predict pathogen-specific MIC values.

    :param device: Torch inference device.
    :param batch_size: Maximum sequences evaluated in one model call.
    :param model: Registered APEX ensemble variant.
    :param models_directory: Optional root used by tests or external installs.
    :raises ValueError: If the model name or batch size is invalid.
    :raises FileNotFoundError: If the complete ensemble is unavailable.
    """

    def __init__(
        self,
        device="cpu",
        batch_size=3000,
        model="default",
        models_directory: str | Path | None = None,
    ):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("APEX batch_size must be a positive integer.")
        self.device = torch.device(device)
        self.model = model
        try:
            variant = APEX_MODEL_VARIANTS[model]
        except KeyError as error:
            raise ValueError(
                f"Unknown APEX model: {model!r}. Available models: "
                f"{sorted(APEX_MODEL_VARIANTS)}."
            ) from error
        self.pathogen_list = tuple(variant.pathogen_list)

        self.max_len = 52  # maximum seq length; 52 = start character + maximum peptide length (50 aa) + end character; longer peptides will be truncated
        self.word2idx, self.idx2word = make_vocab()  # make amino acid vocabulary
        # emb, AAindex_dict = AAindex('./aaindex1.csv', word2idx) #make amino acid embeddings

        # Model resolution
        ## Preserve the historical loader while sharing artifact validation
        _, model_paths = resolve_oracle_model(
            "apex_original",
            model,
            models_directory=models_directory,
        )

        started_at = perf_counter()
        self.APEX_models = []
        for model_path in model_paths:
            loaded_model = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False,
                pickle_module=APEXPickleModule,
            )
            loaded_model.to(self.device).eval()
            self.APEX_models.append(loaded_model)

        self.batch_size = batch_size
        logger.info(
            "Loaded APEX ensemble model=%s models=%s device=%s duration_seconds=%.6f.",
            model,
            len(self.APEX_models),
            self.device,
            perf_counter() - started_at,
        )

    # Use pretrained APEX models to predict species-specific antimicrobial activity (i.e., minimum inhibitory concentration [MIC]; unit: uM)
    # 8 pretrained APEX models are provided, and predictions are averaged
    def predict(self, seq_list, use_tqdm: bool = False):
        """Predict MIC values averaged across the configured ensemble.

        :param seq_list: Peptide strings.
        :param use_tqdm: Display optional model and batch progress bars.
        :return: MIC matrix shaped ``(B, P)``.
        """
        if not seq_list:
            return np.empty((0, len(self.pathogen_list)), dtype=np.float32)
        predictions_by_model: list[np.ndarray] = []

        data_len = len(seq_list)
        num_models = len(self.APEX_models)
        outer_bar = tqdm(total=num_models, desc="Models") if use_tqdm else None

        for ensemble_id in range(num_models):
            apex_model = self.APEX_models[ensemble_id]

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
            model_batches: list[np.ndarray] = []
            for i in batch_iter:
                seq_batch = seq_list[i * self.batch_size : (i + 1) * self.batch_size]
                seq_rep = onehot_encoding(
                    seq_batch, self.max_len, self.word2idx
                )  # make input
                x_sequence = torch.as_tensor(
                    seq_rep,
                    dtype=torch.long,
                    device=self.device,
                )  # (B_model, L)
                with torch.inference_mode():
                    amp_prediction_batch = apex_model(x_sequence)  # (B_model, P)
                AMP_pred_batch = amp_prediction_batch.cpu().numpy()
                AMP_pred_batch = 10 ** (
                    6 - AMP_pred_batch
                )  # transform back to MICs; When training the APEX models, MICs were transformed by: -np.log10(MICs/float(1000000))

                model_batches.append(AMP_pred_batch)

                if inner_bar is not None:
                    inner_bar.update(len(seq_batch))

            predictions_by_model.append(np.concatenate(model_batches, axis=0))

            if inner_bar is not None:
                inner_bar.close()
            if outer_bar is not None:
                outer_bar.update(1)

        if outer_bar is not None:
            outer_bar.close()

        stacked = np.stack(predictions_by_model, axis=0)  # (M, B, P)
        return np.mean(stacked, axis=0)  # (B, P)
