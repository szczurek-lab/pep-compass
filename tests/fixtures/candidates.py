"""Candidate batches with deterministic shapes and field values."""

import torch

from pep_compass.data.optimization import CandidateBatch, TensorField


def candidate_batch() -> CandidateBatch:
    """Return a two-row candidate batch with one aligned score tensor."""
    return CandidateBatch(
        ("AA", "BB"),
        torch.tensor([[0.0, 1.0], [2.0, 3.0]]),  # (B=2, D=2)
        {"score": TensorField(torch.tensor([0.25, 0.75]))},  # (B=2,)
    )
