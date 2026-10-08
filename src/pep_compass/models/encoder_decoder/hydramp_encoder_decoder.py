import os
from typing import Literal

import torch
from einops import rearrange, repeat
from torch import nn

from pep_compass.models.encoder_decoder.encoder_decoder import EncoderDecoder
from pep_compass.models.hydramp.hydramp import HydrAMPDecoder, HydrAMPEncoder
from pep_compass.utils.sequence_utils import to_one_hot, translate_generated_peptide


class HydrAMPEncoderDecoder(EncoderDecoder, nn.Module):
    def __init__(
        self,
        *,
        jacobian_mode: Literal["strict", "approx"] = "approx",
        device: torch.device = "cpu",
        default_condition: torch.Tensor = torch.tensor([1.0, 1.0]),
        temp: float = 1.0,
        jacobian_eps: float = 0.05,
        field_eps: float = 0.05,
    ):
        assert default_condition.ndim == 1, ValueError(
            f"Default condition should be 1D, got {default_condition.ndim}D instead."
        )
        assert default_condition.shape[0] == 2, ValueError(
            f"Default condition should have 2 elements, got {default_condition.shape[0]} instead."
        )
        EncoderDecoder.__init__(
            self,
            jacobian_mode=jacobian_mode,
            jacobian_eps=jacobian_eps,
            field_eps=field_eps,
        )
        nn.Module.__init__(self)

        self.device = device

        self.default_condition = default_condition.to(device)
        self.temp = temp

        self.encoder = HydrAMPEncoder(device=device)
        self.decoder = HydrAMPDecoder(device=device)

        self._load_weights()

    def _load_weights(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = f"{file_dir}/../hydramp/weights"

        if not os.path.exists(weights_dir):
            raise FileNotFoundError(
                f"Weights directory {weights_dir} not found. To get the HydrAMP weights follow the instructtions in README.md under 'Download Model Weights' section."
            )

        self.encoder.load_state_dict(
            torch.load(f"{weights_dir}/encoder_weights.pickle", weights_only=True)
        )
        self.decoder.load_state_dict(
            torch.load(f"{weights_dir}/decoder_weights.pickle", weights_only=True)
        )

    @property
    def latent_dim(self):
        return 64

    @property
    def ambient_dim(self):
        return 525

    def decoder_forward(
        self,
        x: torch.Tensor,
        softmax: bool = True,
        log_softmax: bool = False,
        flatten: bool = True,
    ) -> torch.Tensor:
        assert not (softmax and log_softmax), ValueError(
            "Cannot use both softmax and log_softmax at the same time"
        )
        if x.ndim == 1:
            x = x.unsqueeze(0)

        decoder_input = torch.cat(
            [
                x,
                repeat(
                    self.default_condition.to(self.device), "c -> b c", b=x.shape[0]
                ),
            ],
            dim=-1,
        )
        decoder_output = self.decoder(decoder_input)

        if softmax:
            decoder_output = torch.softmax(decoder_output / self.temp, dim=-1)
        elif log_softmax:
            decoder_output = torch.log_softmax(decoder_output / self.temp, dim=-1)
        if flatten:
            decoder_output = rearrange(decoder_output, "b seq vocab -> b (seq vocab)")
            assert decoder_output.shape[-1] == self.ambient_dim, ValueError(
                f"Decoder output shape is {decoder_output.shape[-1]}, expected {self.ambient_dim}"
            )
        return decoder_output

    def decode_peptides(self, batch: torch.Tensor, batch_size: int = 1) -> list[str]:
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)

        decoded_peptides = []
        for i in range(0, batch.shape[0], batch_size):
            z = batch[i : i + batch_size]
            decoded_logits = self.decoder_forward(z, softmax=False, flatten=False)
            decoded_peptides.extend(
                [
                    translate_generated_peptide(logits.unsqueeze(0))
                    for logits in decoded_logits
                ]
            )

        return decoded_peptides

    def encoder_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.encoder(x)
        return mean, std

    def encode_peptides_with_std(
        self, peptides: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one_hot_peptides = torch.tensor(
            [to_one_hot(peptide) for peptide in peptides], device=self.device
        )
        return self.encoder_forward(one_hot_peptides)

    def encode_peptides(self, peptides: list[str]) -> torch.Tensor:
        one_hot_peptides = torch.tensor(
            [to_one_hot(peptide) for peptide in peptides], device=self.device
        )
        return self.encoder_forward(one_hot_peptides)[0]
