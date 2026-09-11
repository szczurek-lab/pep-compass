"""Tests for shared optimization data contracts."""

import torch

from pep_compass.data.optimization import TensorField
from fixtures.candidates import candidate_batch


def test_candidate_selection_preserves_every_aligned_column() -> None:
    """Selecting a row must select sequences, latents and fields together."""
    selected = candidate_batch().select(torch.tensor([1]))

    assert selected.sequences == ("BB",)
    assert selected.latent_origins.tolist() == [[2.0, 3.0]]
    assert isinstance(selected.fields["score"], TensorField)
    assert selected.fields["score"].values.tolist() == [0.75]


def test_candidate_batch_can_release_transient_fields() -> None:
    """Removing fields must preserve shared sequence and latent storage."""
    batch = candidate_batch()

    reduced = batch.without_fields(("score",))

    assert reduced.sequences == batch.sequences
    assert reduced.latent_origins is batch.latent_origins
    assert reduced.fields == {}
