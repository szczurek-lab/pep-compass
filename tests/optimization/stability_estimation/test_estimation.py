"""Tests for static candidate-batch memory estimates."""

from pep_compass.optimization.stability_estimation import estimate_batch_memory
from fixtures.candidates import candidate_batch


def test_estimate_accounts_for_latents_and_tensor_fields() -> None:
    """Tensor byte counts must follow shapes and dtypes exactly."""
    estimate = estimate_batch_memory(candidate_batch())

    assert estimate.latent_bytes == 2 * 2 * 4
    assert estimate.field_bytes == 2 * 4
    assert estimate.total_bytes >= estimate.latent_bytes + estimate.field_bytes
