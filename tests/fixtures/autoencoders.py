"""Small deterministic autoencoders with analytically known outputs."""

from __future__ import annotations

import torch


class MockAutoencoder:
    """Encode sequences by index and return diagonal decoder Jacobians."""

    latent_dim = 2
    ambient_dim = 2

    def encode_peptides(self, sequences: list[str]) -> torch.Tensor:
        """Return deterministic two-dimensional latent positions."""
        return torch.arange(
            len(sequences) * 2,
            dtype=torch.float32,
        ).reshape(len(sequences), 2)  # (B, 2)

    def decoder_jacobian(self, latent_positions: torch.Tensor) -> torch.Tensor:
        """Return diagonal Jacobians derived from latent coordinates."""
        return torch.stack(
            [
                torch.diag(position + 1.0)
                for position in latent_positions
            ]
        )  # (B, 2, 2)
