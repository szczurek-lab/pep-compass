"""Geometry derived from an autoencoder decoder Jacobian."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pep_compass.autoencoder.base import Autoencoder


@dataclass(frozen=True, slots=True)
class TangentDecomposition:
    """Store a batched compact SVD of decoder Jacobians.

    :param left_vectors: Ambient-space singular vectors ``(B, A, K)``.
    :param singular_values: Singular values ``(B, K)``.
    :param right_vectors: Latent-space right singular vectors ``(B, K, D)``.
    """

    left_vectors: torch.Tensor
    singular_values: torch.Tensor
    right_vectors: torch.Tensor


@dataclass(frozen=True, slots=True)
class StableTangentGeometry:
    """Store one batched decoder geometry shared by a SORBES step.

    :param decomposition: Compact SVD of decoder Jacobians.
    :param kappa: Squared singular-value cutoff from the SORBES notation.
    :param active_mask: Stable directions satisfying ``S**2 > kappa`` with
        shape ``(B, K)``.

    The object owns no copied matrices. Consumers use views of the same SVD
    tensors, so SORBES integration and mutation enumeration cannot silently
    derive geometry from different Jacobians.
    """

    decomposition: TangentDecomposition
    kappa: float
    active_mask: torch.Tensor

    @property
    def left_vectors(self) -> torch.Tensor:
        """Return ambient singular vectors ``(B, A, K)``."""
        return self.decomposition.left_vectors

    @property
    def singular_values(self) -> torch.Tensor:
        """Return singular values ``(B, K)``."""
        return self.decomposition.singular_values

    @property
    def right_vectors(self) -> torch.Tensor:
        """Return latent right singular vectors ``(B, K, D)``."""
        return self.decomposition.right_vectors

    def project_ambient_to_active_latent(
        self, ambient_vectors: torch.Tensor
    ) -> torch.Tensor:
        """Apply the truncated Jacobian pseudoinverse without materialising it.

        :param ambient_vectors: Ambient vectors ``(B, A)``.
        :return: Active latent projections ``(B, D)``.
        """
        coefficients = torch.bmm(
            self.left_vectors.transpose(1, 2), ambient_vectors.unsqueeze(-1)
        ).squeeze(-1)  # (B, K)
        inverse = torch.where(
            self.active_mask,
            self.singular_values.reciprocal(),
            torch.zeros_like(self.singular_values),
        )  # (B, K)
        weighted = coefficients * inverse  # (B, K)
        return torch.bmm(
            self.right_vectors.transpose(1, 2), weighted.unsqueeze(-1)
        ).squeeze(-1)  # (B, D)

    def select(self, index: int) -> "StableTangentGeometry":
        """Return a one-row view without copying SVD tensor storage."""
        decomposition = TangentDecomposition(
            self.left_vectors[index : index + 1],
            self.singular_values[index : index + 1],
            self.right_vectors[index : index + 1],
        )
        return StableTangentGeometry(
            decomposition,
            self.kappa,
            self.active_mask[index : index + 1],
        )


def compute_tangent_decomposition(
    autoencoder: Autoencoder,
    latent_positions: torch.Tensor,
) -> TangentDecomposition:
    """Compute compact decoder-Jacobian SVDs for latent positions.

    :param autoencoder: Autoencoder providing batched decoder Jacobians.
    :param latent_positions: One latent vector ``(D,)`` or batch ``(B, D)``.
    :return: Named batched tangent decomposition.
    """
    if latent_positions.ndim == 1:
        latent_positions = latent_positions.unsqueeze(0)  # (1, D)
    decoder_jacobians = autoencoder.decoder_jacobian(latent_positions)  # (B, A, D)
    left, singular, right = torch.linalg.svd(
        decoder_jacobians,
        full_matrices=False,
    )
    return TangentDecomposition(left, singular, right)


def compute_stable_tangent_geometry(
    autoencoder: Autoencoder,
    latent_positions: torch.Tensor,
    *,
    kappa: float,
) -> StableTangentGeometry:
    r"""Compute the reusable :math:`\kappa`-stable geometry for one step.

    :param autoencoder: Autoencoder providing decoder Jacobians.
    :param latent_positions: Latent batch ``(B, D)``.
    :param kappa: Non-negative squared singular-value cutoff.
    :return: Batched geometry retained until local candidate filtering ends.
    :raises ValueError: If ``kappa`` is negative.
    """
    if kappa < 0:
        raise ValueError("SORBES kappa must be non-negative.")
    decomposition = compute_tangent_decomposition(autoencoder, latent_positions)
    active_mask = decomposition.singular_values.square() > kappa  # (B, K)
    return StableTangentGeometry(decomposition, kappa, active_mask)
