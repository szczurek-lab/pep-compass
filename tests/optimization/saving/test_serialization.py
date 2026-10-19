"""Exact scientific serialization, masks, aliases and registered extensions."""

from dataclasses import dataclass

import pytest
import torch

from pep_compass.data_schemas.candidates import (
    CandidateBatch,
    ObjectField,
    OptionalField,
    SharedField,
    TensorField,
)
from pep_compass.optimization.saving.serialization import dumps, loads


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.int64,
        torch.bool,
    ],
)
@pytest.mark.parametrize("shape", [(), (0, 2), (2, 3)])
def test_tensor_codec_is_exact(dtype, shape):
    value = torch.ones(shape, dtype=dtype)
    restored = loads(dumps(value))
    assert restored.dtype == value.dtype
    assert restored.shape == value.shape
    assert torch.equal(restored, value)


def test_candidate_fields_roundtrip_without_mask_or_dtype_loss():
    batch = CandidateBatch(
        ["AA", "AC"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
        {
            "optional": OptionalField(
                TensorField(torch.tensor([5, 6])), torch.tensor([True, False])
            ),
            "objects": ObjectField([{"a": 1}, None]),
            "shared": SharedField("value"),
        },
    )
    restored = loads(dumps(batch))
    assert torch.equal(restored.latent_origins, batch.latent_origins)
    assert restored.latent_origins.dtype == torch.float64
    assert restored.fields["optional"].valid.tolist() == [True, False]
    assert restored.fields["objects"].values == ({"a": 1}, None)


def test_serialization_preserves_mutation_pool_aliases():
    options = {0: [1, 2], 1: [3]}
    batch = CandidateBatch(
        ["AA", "AC"],
        torch.ones(2, 2),
        {
            "mutation.options": ObjectField([options, options]),
            "a": SharedField("first"),
            "b": SharedField("second"),
        },
    )
    restored = loads(dumps(batch))
    assert (
        restored.fields["mutation.options"].values[0]
        is restored.fields["mutation.options"].values[1]
    )
    assert restored.fields["a"].value == "first"
    assert restored.fields["b"].value == "second"


def test_unsupported_payloads_are_rejected_without_dynamic_registration():
    @dataclass
    class Trace:
        values: torch.Tensor

    with pytest.raises(TypeError, match="No run-data codec"):
        dumps(Trace(torch.arange(6, dtype=torch.float64).reshape(2, 3)))
