"""Tests for geometry derived from decoder Jacobians."""

import torch

from pep_compass.autoencoder.geometry import (
    compute_stable_tangent_geometry,
    compute_tangent_decomposition,
)
from fixtures.autoencoders import MockAutoencoder


def test_tangent_decomposition_batches_single_latent_position() -> None:
    """A one-dimensional latent position must produce a one-row decomposition."""
    result = compute_tangent_decomposition(
        MockAutoencoder(),
        torch.tensor([1.0, 3.0]),  # (D=2,)
    )

    assert result.left_vectors.shape == (1, 2, 2)
    assert result.singular_values.shape == (1, 2)
    assert result.right_vectors.shape == (1, 2, 2)
    assert result.singular_values.tolist() == [[4.0, 2.0]]


def test_stable_geometry_reuses_svd_for_truncated_projection() -> None:
    """Projection must use the configured squared singular-value cutoff."""
    geometry = compute_stable_tangent_geometry(
        MockAutoencoder(),
        torch.tensor([[1.0, 3.0]]),  # (B=1, D=2)
        kappa=9.0,
    )

    projected = geometry.project_ambient_to_active_latent(
        torch.tensor([[8.0, 8.0]])
    )  # (B=1, D=2)

    assert geometry.active_mask.tolist() == [[True, False]]
    assert torch.allclose(projected, torch.tensor([[0.0, 2.0]]), atol=1e-6)
