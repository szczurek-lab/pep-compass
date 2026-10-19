"""Small direct resource and pipeline constructors for integration tests."""

import torch

from pep_compass import Pipeline
from pep_compass.autoencoder.base import Autoencoder
from pep_compass.steps.filters import CandidateSubset
from pep_compass.steps.oracles import Hydrophobicity
from pep_compass.utils.configuration import Configurable


class FixtureAutoencoder(Autoencoder):
    """Two-dimensional encoder without files, downloads or random weights."""

    latent_dim = 2
    ambient_dim = 2

    def __init__(self, device="cpu"):
        super().__init__(jacobian_eps=0.01, field_eps=0.01)
        self.device = device

    def encode_peptides(self, peptides):
        return torch.tensor(
            [[float(len(sequence)), 1.0] for sequence in peptides], device=self.device
        )  # (B, 2)

    def decode_peptides(self, latent_positions):
        return ["AA" for _ in range(len(latent_positions))]

    def decoder_forward(self, latent_positions):
        return latent_positions  # (B, 2)

    def encoder_forward(self, encoded_sequences):
        return encoded_sequences  # (B, 2)


class FixtureResource(Configurable):
    """Declare the deterministic encoder without allocating model state."""

    def __init__(self, device="cpu"):
        self._configure({"device": device})

    def build(self, *, device=None):
        return FixtureAutoencoder(device=device or self.configuration["device"])


def create(*, seed=7, count=2, device="cpu"):
    pipeline = Pipeline(
        [CandidateSubset(count=count), Hydrophobicity()],
        autoencoder=FixtureResource(device),
    )
    pipeline.set_runner_settings(execution={"seed": seed, "oracle_calls": 3})
    return pipeline
