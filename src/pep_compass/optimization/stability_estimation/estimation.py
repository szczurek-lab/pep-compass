"""Static byte-size estimates for optimization data structures."""

from __future__ import annotations

from dataclasses import dataclass
import sys

import torch

from pep_compass.data.optimization import (
    CandidateBatch,
    ObjectField,
    OptionalField,
    SharedField,
    TensorField,
)


@dataclass(frozen=True, slots=True)
class BatchMemoryEstimate:
    """Describe bytes directly retained by a candidate batch.

    :param sequence_bytes: Estimated Python string storage.
    :param latent_bytes: Storage occupied by latent-origin tensors.
    :param field_bytes: Storage occupied by candidate fields.
    :param total_bytes: Sum of the estimated retained storage.
    """

    sequence_bytes: int
    latent_bytes: int
    field_bytes: int
    total_bytes: int


def _tensor_bytes(value: torch.Tensor) -> int:
    """Return tensor storage required by its logical elements."""
    return value.numel() * value.element_size()


def estimate_batch_memory(batch: CandidateBatch) -> BatchMemoryEstimate:
    """Estimate storage directly retained by a candidate batch.

    The estimate excludes allocator fragmentation, tensor storage aliases,
    model weights, CUDA workspaces and interpreter container overhead. It is
    intended for fast relative diagnostics at step boundaries.

    :param batch: Candidate batch to inspect.
    :return: Estimated byte counts grouped by storage responsibility.
    """
    sequence_bytes = sum(sys.getsizeof(sequence) for sequence in batch.sequences)
    latent_bytes = _tensor_bytes(batch.latent_origins)
    field_bytes = 0
    for field in batch.fields.values():
        if isinstance(field, TensorField):
            field_bytes += _tensor_bytes(field.values)
        elif isinstance(field, OptionalField):
            field_bytes += _tensor_bytes(field.valid)
            if isinstance(field.values, TensorField):
                field_bytes += _tensor_bytes(field.values.values)
        elif isinstance(field, ObjectField):
            field_bytes += sum(sys.getsizeof(value) for value in field.values)
        elif isinstance(field, SharedField):
            field_bytes += sys.getsizeof(field.value)
    return BatchMemoryEstimate(
        sequence_bytes=sequence_bytes,
        latent_bytes=latent_bytes,
        field_bytes=field_bytes,
        total_bytes=sequence_bytes + latent_bytes + field_bytes,
    )
