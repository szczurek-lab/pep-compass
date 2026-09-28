
import numpy as np
import torch
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation

from pep_compass.models.apex.APEX_predictor import PredictorAPEX
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import \
    HydrAMPEncoderDecoder


class APEXBlackBox(AbstractBlackBox):
    def __init__(
        self,
        *,
        mic_aggregate: str = "mean",
        mic_bacteria: str | list = "all",
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        
        self.apex_predictor = PredictorAPEX(device=device)
        
        if mic_aggregate == "max":
            mic_aggregate_func = lambda x: np.max(x, axis=1)
        elif mic_aggregate == "mean":
            mic_aggregate_func = lambda x: np.mean(x, axis=1)
        
        if mic_bacteria == "all":
            mic_bacteria_func = lambda x: x
        elif isinstance(mic_bacteria, list):
            mic_bacteria_func = lambda x: x[:, mic_bacteria]
        
        self.peptide_scorer = lambda x: mic_aggregate_func(np.log2(mic_bacteria_func(self.apex_predictor.predict(x))))

        self.maximize = False

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="APEX",
            max_sequence_length=25,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=list("ACDEFGHIKLMNPQRSTVWY"),
            log_transform_recommended=False,
            discrete=True,
            padding_token=" ",
        )

    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        sequences = ["".join(seq) for seq in x]
        predictions = self.peptide_scorer(sequences)

        return predictions.reshape(-1, 1)

class HydrAMPAPEXBlackBox(AbstractBlackBox):
    def __init__(
        self,
        *,
        mic_aggregate: str = "mean",
        mic_bacteria: str | list = "all",
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
        device: str = "cpu",
        jacobian_eps: float,
        field_eps: float,
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        
        self.apex_predictor = PredictorAPEX(device=device, batch_size=1)
        
        if mic_aggregate == "max":
            mic_aggregate_func = lambda x: np.max(x, axis=1)
        elif mic_aggregate == "mean":
            mic_aggregate_func = lambda x: np.mean(x, axis=1)
        
        if mic_bacteria == "all":
            mic_bacterias_func = lambda x: x
        elif isinstance(mic_bacteria, list):
            mic_bacterias_func = lambda x: x[:, mic_bacteria]
        
        self.peptide_scorer = lambda x: mic_aggregate_func(np.log2(mic_bacterias_func(self.apex_predictor.predict(x))))
        
        self.encoder_decoder = HydrAMPEncoderDecoder(
            device=device,
            default_condition=torch.tensor([1, 1], device=device),
            temp=1,
            jacobian_mode="approx",
            jacobian_eps=jacobian_eps,
            field_eps=field_eps,
        )

        self.cache = [] 

        self.shift = 0.0
        
    def set_shift(self, shift: float):
        self.shift = shift

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="APEX",
            max_sequence_length=25,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=list("ACDEFGHIKLMNPQRSTVWY"),
            log_transform_recommended=False,
            discrete=False,
            padding_token=" ",
        )

    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        x_tensor = torch.tensor(x, device=self.encoder_decoder.device)
        decoded_peptides = self.encoder_decoder.decode_peptides(x_tensor)
        predictions = self.peptide_scorer(decoded_peptides)
        
        if context is not None and isinstance(context, dict):
            context['sequences'] = decoded_peptides
        
        for i, seq in enumerate(decoded_peptides):
            self.cache.append((x_tensor[i].tolist(), seq, predictions[i].item()))

        return predictions.reshape(-1, 1) + self.shift  # Reshape to match the expected output shape

    def clear_cache(self):
        self.cache = []
