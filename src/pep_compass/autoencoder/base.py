"""Public autoencoder contract used by PepCompass computations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch


class Autoencoder(ABC):
    """Encode peptides and expose differentiable decoder geometry."""

    def __init__(
        self,
        *,
        jacobian_mode: Literal["strict", "approx"] = "strict",
        jacobian_eps: float,
        field_eps: float,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param jacobian_mode: Jacobian mode.
        :type jacobian_mode: Literal['strict', 'approx']
        :param jacobian_eps: Jacobian eps.
        :type jacobian_eps: float
        :param field_eps: Field eps.
        :type field_eps: float
        :rtype: None
        """
        if jacobian_mode not in {"strict", "approx"}:
            raise ValueError(f"Unknown Jacobian mode: {jacobian_mode}")
        from pep_compass.utils.validation import validate_positive

        validate_positive(jacobian_eps, "autoencoder.jacobian_eps")
        validate_positive(field_eps, "autoencoder.field_eps")
        self.jacobian_mode = jacobian_mode
        self.jacobian_eps = jacobian_eps
        self.field_eps = field_eps

    @abstractmethod
    def decoder_forward(self, latent_positions: torch.Tensor) -> torch.Tensor:
        """Decode independent latent rows into an ambient tensor (B, A).

        :param latent_positions: Latent coordinates with shape (B, D).
        :type latent_positions: torch.Tensor
        :return: Result of this operation.
        :rtype: torch.Tensor
        """

    @abstractmethod
    def encoder_forward(
        self, encoded_sequences: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Encode model-specific sequence tensors.

        :param encoded_sequences: Encoded sequences.
        :type encoded_sequences: torch.Tensor
        :return: Result of this operation.
        :rtype: tuple[torch.Tensor, torch.Tensor] | torch.Tensor
        """

    @abstractmethod
    def encode_peptides(self, peptides: list[str]) -> torch.Tensor:
        """Encode peptide strings into latent means ``(B, D)``.

        :param peptides: Peptide strings to encode.
        :type peptides: list[str]
        :return: Result of this operation.
        :rtype: torch.Tensor
        """

    @abstractmethod
    def decode_peptides(self, latent_positions: torch.Tensor) -> list[str]:
        """Decode latent rows into peptide strings for walker execution.

        :param latent_positions: Latent batch with shape (B, D).
        :return: One decoded sequence per input row.
        :raises NotImplementedError: If string decoding is unsupported.

        :type latent_positions: torch.Tensor
        :rtype: list[str]
        """
        raise NotImplementedError(
            "This autoencoder does not implement peptide decoding."
        )

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Return latent dimension ``D``.

        :return: Result of this operation.
        :rtype: int
        """

    @property
    @abstractmethod
    def ambient_dim(self) -> int:
        """Return flattened decoder-output dimension ``A``.

        :return: Result of this operation.
        :rtype: int
        """

    def decoder_jacobian(self, latent_positions: torch.Tensor) -> torch.Tensor:
        """Return decoder Jacobians with shape ``(B, A, D)``.

        :param latent_positions: Latent coordinates with shape (B, D).
        :type latent_positions: torch.Tensor
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        if latent_positions.ndim != 2:
            raise ValueError("Latent positions must have shape (B, D).")
        if self.jacobian_mode == "strict":
            return self._decoder_jacobian_strict(latent_positions)
        return self._decoder_jacobian_approx(latent_positions)

    def _decoder_jacobian_strict(self, latent_positions: torch.Tensor) -> torch.Tensor:
        """Compute exact per-sample decoder Jacobians through autograd.

        :param latent_positions: Latent coordinates with shape (B, D).
        :type latent_positions: torch.Tensor
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        # Row-separable decoder differentiation
        ## REMARK: Summing the output batch removes the redundant output-batch
        ## axis. Independent decoder rows then produce their own Jacobian blocks,
        ## without allocating the full (B, A, B, D) cross-sample Jacobian.
        jacobian = torch.autograd.functional.jacobian(
            lambda positions: self.decoder_forward(positions).sum(dim=0),
            latent_positions,
        )  # (A, B, D)
        return jacobian.permute(1, 0, 2)  # (B, A, D)

    def _decoder_jacobian_approx(self, latent_positions: torch.Tensor) -> torch.Tensor:
        """Approximate batched decoder Jacobians by forward differences.

        :param latent_positions: Latent coordinates with shape (B, D).
        :type latent_positions: torch.Tensor
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
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
        decoder_input = (latent_positions[:, None, :] + delta[None, :, :]).reshape(
            -1, self.latent_dim
        )  # (B * (D + 1), D)
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
        """Approximate the second directional derivative of the decoder.

        :param latent_point: Latent point.
        :type latent_point: torch.Tensor
        :param direction: Objective ordering, minimize or maximize.
        :type direction: torch.Tensor
        :param ambient_point: Ambient point.
        :type ambient_point: torch.Tensor | None
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        center = (
            self.decoder_forward(latent_point)
            if ambient_point is None
            else ambient_point
        )
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
        """Approximate a mixed decoder derivative by central differences.

        :param latent_point: Latent point.
        :type latent_point: torch.Tensor
        :param vector1: Vector1.
        :type vector1: torch.Tensor
        :param vector2: Vector2.
        :type vector2: torch.Tensor
        :param eps: Eps.
        :type eps: float
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        return (
            self.decoder_forward(latent_point + eps * vector1 + eps * vector2)
            - self.decoder_forward(latent_point - eps * vector1 + eps * vector2)
            - self.decoder_forward(latent_point + eps * vector1 - eps * vector2)
            + self.decoder_forward(latent_point - eps * vector1 - eps * vector2)
        ) / (4 * eps**2)
