"""HydrAMP implementation of the PepCompass autoencoder contract."""

import torch
from typing import Literal
from torch import nn
from pathlib import Path
from time import perf_counter

from pep_compass.autoencoder.base import Autoencoder
from pep_compass.autoencoder.strategies.hydramp.architecture import (
    HydrAMPDecoder,
    HydrAMPEncoder,
)
from pep_compass.utils.sequence_utils import to_one_hot, translate_generated_peptide
from pep_compass.utils.logger import get_custom_logger
from pep_compass.registry import model_catalog
from einops import repeat, rearrange

logger = get_custom_logger(__name__)


class HydrampAutoencoder(Autoencoder, nn.Module):
    """Load and execute the HydrAMP peptide autoencoder."""
    def __init__(
        self,
        *,
        jacobian_mode: Literal["strict", "approx"] = "strict",
        device: torch.device = "cpu",
        default_condition: torch.Tensor | None = None,
        temp: float = 1.0,
        jacobian_eps: float,
        field_eps: float,
        model_name: str,
    ):
        if default_condition is None:
            default_condition = torch.tensor([1.0, 1.0])
        assert default_condition.ndim == 1, ValueError(
            f"Default condition should be 1D, got {default_condition.ndim}D instead."
        )
        assert default_condition.shape[0] == 2, ValueError(
            f"Default condition should have 2 elements, got {default_condition.shape[0]} instead."
        )
        Autoencoder.__init__(
            self,
            jacobian_mode=jacobian_mode,
            jacobian_eps=jacobian_eps,
            field_eps=field_eps,
        )
        nn.Module.__init__(self)

        self.device = device

        self.default_condition = default_condition.to(device)
        self.temp = temp
        self.model_name = model_name

        self.encoder = HydrAMPEncoder(device=device)
        self.decoder = HydrAMPDecoder(device=device)

        self._load_weights()

    def _load_weights(self) -> None:
        """Load state dictionaries for the selected bundled HydrAMP model."""
        # Direct adapter use bypasses the runtime bootstrap registration path.
        if self.model_name not in model_catalog.names("autoencoder.hydramp"):
            from pep_compass.autoencoder.strategies.hydramp import register_hydramp

            register_hydramp()
        started_at = perf_counter()
        weights_dir = Path(__file__).parent / "models" / self.model_name
        if not weights_dir.is_dir():
            raise FileNotFoundError(
                f"HydrAMP weights directory does not exist: {weights_dir}"
            )
        _, paths = model_catalog.resolve(
            "autoencoder.hydramp",
            self.model_name,
            root=Path(__file__).parent / "models",
        )
        encoder_path, decoder_path = paths

        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=self.device, weights_only=True)
        )
        self.decoder.load_state_dict(
            torch.load(decoder_path, map_location=self.device, weights_only=True)
        )
        logger.info(
            "Loaded HydrAMP weights directory=%s duration_seconds=%.6f.",
            encoder_path.parent,
            perf_counter() - started_at,
        )

    # model properties
    @property
    def latent_dim(self) -> int:
        """Return the HydrAMP latent dimension."""
        return self.encoder.mean_linear.out_features

    @property
    def ambient_dim(self) -> int:
        """Return the flattened decoder probability dimension."""
        return self.decoder.gru.output_len * self.decoder.dense.out_features


    # main functionality 
    def decoder_forward(
        self,
        x: torch.Tensor,
        softmax: bool = True,
        log_softmax: bool = False,
        flatten: bool = True,
    ) -> torch.Tensor:
        """Decode latent positions into token logits or probabilities.

        :param x: Latent positions ``(B, D)`` or ``(D,)``.
        :param softmax: Return token probabilities.
        :param log_softmax: Return token log-probabilities.
        :param flatten: Flatten sequence and vocabulary dimensions.
        :return: Decoder output ``(B, L * V)`` or ``(B, L, V)``.
        """
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
        )  # (B, D + C)
        decoder_output = self.decoder(decoder_input)  # (B, L, V)

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
        """Decode latent positions into peptide strings in bounded batches."""
        started_at = perf_counter()
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

        logger.debug(
            "Decoded HydrAMP peptides candidates=%s batch_size=%s duration_seconds=%.6f.",
            batch.shape[0],
            batch_size,
            perf_counter() - started_at,
        )
        return decoded_peptides

    def encoder_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent mean and scale for encoded sequence tokens."""
        mean, std = self.encoder(x)
        return mean, std

    def encode_peptides_with_std(
        self, peptides: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode peptide strings into latent means and scales."""
        one_hot_peptides = torch.tensor(
            [to_one_hot(peptide) for peptide in peptides], device=self.device
        )
        return self.encoder_forward(one_hot_peptides)

    def encode_peptides(self, peptides: list[str]) -> torch.Tensor:
        """Encode peptide strings into latent means ``(B, D)``."""
        started_at = perf_counter()
        one_hot_peptides = torch.tensor(
            [to_one_hot(peptide) for peptide in peptides], device=self.device
        )
        latent_means = self.encoder_forward(one_hot_peptides)[0]  # (B, D)
        logger.debug(
            "Encoded HydrAMP peptides candidates=%s latent_dimension=%s "
            "duration_seconds=%.6f.",
            len(peptides),
            latent_means.shape[-1],
            perf_counter() - started_at,
        )
        return latent_means
