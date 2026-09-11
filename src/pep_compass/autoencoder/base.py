"""Public autoencoder contract used by PepCompass computations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from einops import rearrange


class Autoencoder(ABC):
    """Encode peptides and expose differentiable decoder geometry."""

    def __init__(
        self,
        *,
        jacobian_mode: Literal["strict", "approx"] = "strict",
        jacobian_eps: float,
        field_eps: float,
    ) -> None:
        if jacobian_mode not in {"strict", "approx"}:
            raise ValueError(f"Unknown Jacobian mode: {jacobian_mode}")
        if jacobian_eps <= 0 or field_eps <= 0:
            raise ValueError("Finite-difference epsilons must be positive.")
        self.jacobian_mode = jacobian_mode
        self.jacobian_eps = jacobian_eps
        self.field_eps = field_eps

    @abstractmethod
    def decoder_forward(self, latent_positions: torch.Tensor) -> torch.Tensor:
        """Decode latent positions into an ambient representation."""

    @abstractmethod
    def encoder_forward(self, encoded_sequences: torch.Tensor):
        """Encode model-specific sequence tensors."""

    @abstractmethod
    def encode_peptides(self, peptides: list[str]) -> torch.Tensor:
        """Encode peptide strings into latent means ``(B, D)``."""

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Return latent dimension ``D``."""

    @property
    @abstractmethod
    def ambient_dim(self) -> int:
        """Return flattened decoder-output dimension ``A``."""

    def decoder_jacobian(self, latent_positions: torch.Tensor) -> torch.Tensor:
        """Return decoder Jacobians with shape ``(B, A, D)``."""
        if latent_positions.ndim != 2:
            raise ValueError("Latent positions must have shape (B, D).")
        if self.jacobian_mode == "strict":
            return self._decoder_jacobian_strict(latent_positions)
        return self._decoder_jacobian_approx(latent_positions)

    def _decoder_jacobian_strict(self, latent_positions: torch.Tensor) -> torch.Tensor:
        """Compute exact per-sample decoder Jacobians through autograd."""
        jacobian = torch.autograd.functional.jacobian(
            self.decoder_forward,
            latent_positions,
        )[0]
        return rearrange(jacobian, "a b d -> b a d")  # (B, A, D)

    def _decoder_jacobian_approx(self, latent_positions: torch.Tensor) -> torch.Tensor:
        """Approximate batched decoder Jacobians by forward differences."""
        delta = torch.cat(
            [
                torch.eye(
                    self.latent_dim,
                    device=latent_positions.device,
                    dtype=latent_positions.dtype,
                )
                * self.jacobian_eps,
                torch.zeros(
                    (1, self.latent_dim),
                    device=latent_positions.device,
                    dtype=latent_positions.dtype,
                ),
            ],
            dim=0,
        )  # (D + 1, D)
        decoder_input = (
            latent_positions[:, None, :] + delta[None, :, :]
        ).reshape(-1, self.latent_dim)  # (B * (D + 1), D)
        with torch.no_grad():
            decoder_output = self.decoder_forward(decoder_input)
        decoder_output = decoder_output.reshape(
            latent_positions.shape[0],
            self.latent_dim + 1,
            -1,
        )  # (B, D + 1, A)
        jacobian = (
            decoder_output[:, :-1, :] - decoder_output[:, [-1], :]
        ) / self.jacobian_eps  # (B, D, A)
        return jacobian.transpose(1, 2)  # (B, A, D)

    def field_derivative(
        self,
        latent_point: torch.Tensor,
        direction: torch.Tensor,
        *,
        ambient_point: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Approximate the second directional derivative of the decoder."""
        center = self.decoder_forward(latent_point) if ambient_point is None else ambient_point
        right = self.decoder_forward(latent_point + self.field_eps * direction)
        left = self.decoder_forward(latent_point - self.field_eps * direction)
        return (right + left - 2 * center) / self.field_eps**2

    def get_ambient_covariant_derivative(
        self,
        latent_point: torch.Tensor,
        vector1: torch.Tensor,
        vector2: torch.Tensor,
        *,
        eps: float = 0.1,
    ) -> torch.Tensor:
        """Approximate a mixed decoder derivative by central differences."""
        return (
            self.decoder_forward(latent_point + eps * vector1 + eps * vector2)
            - self.decoder_forward(latent_point - eps * vector1 + eps * vector2)
            - self.decoder_forward(latent_point + eps * vector1 - eps * vector2)
            + self.decoder_forward(latent_point - eps * vector1 - eps * vector2)
        ) / (4 * eps**2)
