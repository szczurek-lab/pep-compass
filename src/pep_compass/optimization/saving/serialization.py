"""Lossless versioned codecs for run data; no arbitrary object unpickling."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np
import torch


def _payload_types() -> dict[str, type]:
    """Return the fixed data-format types; algorithm construction is unrelated.

    REMARK: The codec accepts only data containers explicitly supported by this
    file. Persisted data cannot import or construct arbitrary Python targets.

    :return: Result of this operation.
    :rtype: dict[str, type]
    """
    from pep_compass.autoencoder.geometry import (
        StableTangentGeometry,
        TangentDecomposition,
    )
    from pep_compass.autoencoder.subriemannian import PointGeometry
    from pep_compass.data_schemas.candidates import (
        CandidateBatch,
        ObjectField,
        OptionalField,
        SharedField,
        TensorField,
    )
    from pep_compass.data_schemas.execution import OptimizationLimits
    from pep_compass.optimization.engine.execution.state import TrustRegionState
    from pep_compass.optimization.tracking import ExecutionScope
    from pep_compass.steps.mutations.strategies.mutang.schema import (
        MutationOption,
        SelectedDirections,
    )

    declarations = {
        kind.__name__: kind
        for kind in (
            CandidateBatch,
            ObjectField,
            OptionalField,
            SharedField,
            TensorField,
            StableTangentGeometry,
            TangentDecomposition,
            PointGeometry,
            OptimizationLimits,
            TrustRegionState,
            ExecutionScope,
            MutationOption,
            SelectedDirections,
        )
    }
    return declarations


def encode(value: object) -> object:
    """Encode typed values while preserving shared mutation-pool containers.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: object
    """
    return _encode(value, {})


def _encode(value: object, memo: dict[int, tuple[int, object]]) -> object:
    """Convert supported values to a lossless JSON-shaped versioned payload.

    :param value: Primitive, nested containers or an explicitly supported value.
    :return: Typed metadata and raw blocks preserving tensor dtype and dimensions.
    :raises TypeError: If a value has no supported scientific codec.

    :type value: object
    :param memo: Memo.
    :type memo: dict[int, tuple[int, object]]
    :rtype: object
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()  # (N_bytes,)
        return {
            "__type__": "tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data": raw,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object arrays cannot be persisted as run payloads.")
        raw = array.tobytes()
        return {
            "__type__": "ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": raw,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _encode(value.item(), memo)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return (
            value
            if math.isfinite(value)
            else {"__type__": "float", "value": repr(value)}
        )
    if isinstance(value, Mapping):
        if id(value) in memo:
            return {"__type__": "reference", "identity": memo[id(value)][0]}
        identity = len(memo)
        memo[id(value)] = (identity, value)
        return {
            "__type__": "mapping",
            "identity": identity,
            "items": [
                [_encode(key, memo), _encode(item, memo)] for key, item in value.items()
            ],
        }
    if isinstance(value, (tuple, list)):
        if isinstance(value, list):
            if id(value) in memo:
                return {"__type__": "reference", "identity": memo[id(value)][0]}
            identity = len(memo)
            memo[id(value)] = (identity, value)
        else:
            identity = None
        return {
            "__type__": "tuple" if isinstance(value, tuple) else "list",
            "identity": identity,
            "items": [_encode(item, memo) for item in value],
        }
    types = _payload_types()
    from pep_compass.autoencoder.subriemannian import SubRiemannianTangentSpace
    from pep_compass.data_schemas.candidates import CandidateBatch

    if isinstance(value, CandidateBatch):
        return {
            "__type__": "CandidateBatch",
            "values": _encode(
                {
                    "sequences": value.sequences,
                    "latent_origins": value.latent_origins,
                    "fields": value.fields,
                },
                memo,
            ),
        }
    if isinstance(value, SubRiemannianTangentSpace):
        return {
            "__type__": "tangent_space",
            "values": _encode(
                {
                    "U": value.U,
                    "S": value.S,
                    "V": value.V,
                    "horizontal_threshold": value.horizontal_threshold,
                },
                memo,
            ),
        }
    codec_name = next(
        (name for name, kind in types.items() if kind is type(value)), None
    )
    if is_dataclass(value) and codec_name is not None:
        return {
            "__type__": codec_name,
            "values": _encode(
                {field.name: getattr(value, field.name) for field in fields(value)},
                memo,
            ),
        }
    raise TypeError(
        f"No run-data codec for {type(value).__module__}.{type(value).__qualname__}."
    )


def decode(value: object) -> object:
    """Restore only supported codecs, including shared container references.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: object
    """
    return _decode(value, {})


def _decode(value: object, memo: dict[int, object]) -> object:
    """Restore supported values without importing targets supplied by data.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :param memo: Memo.
    :type memo: dict[int, object]
    :return: Result of this operation.
    :rtype: object
    """
    if not isinstance(value, dict) or "__type__" not in value:
        return value
    kind = value["__type__"]
    if kind in {"tensor", "ndarray"}:
        raw = (
            value["data"]
            if isinstance(value["data"], bytes)
            else base64.b64decode(value["data"], validate=True)
        )
        if hashlib.sha256(raw).hexdigest() != value["sha256"]:
            raise ValueError("Tensor payload checksum mismatch.")
        if kind == "ndarray":
            return (
                np.frombuffer(raw, dtype=np.dtype(value["dtype"]))
                .copy()
                .reshape(value["shape"])
            )
        dtype = getattr(torch, value["dtype"], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError("Unknown tensor dtype.")
        tensor = (
            torch.frombuffer(bytearray(raw), dtype=dtype).clone()
            if raw
            else torch.empty(0, dtype=dtype)
        )
        return tensor.reshape(value["shape"])  # (*stored_shape,)
    if kind == "float":
        return float(value["value"])
    if kind == "reference":
        return memo[value["identity"]]
    if kind == "mapping":
        result = {}
        memo[value["identity"]] = result
        for key, item in value["items"]:
            result[_decode(key, memo)] = _decode(item, memo)
        return result
    if kind in {"tuple", "list"}:
        values = []
        if kind == "list":
            memo[value["identity"]] = values
        values.extend(_decode(item, memo) for item in value["items"])
        return tuple(values) if kind == "tuple" else values
    values = _decode(value["values"], memo)
    if kind == "tangent_space":
        from pep_compass.autoencoder.subriemannian import SubRiemannianTangentSpace

        return SubRiemannianTangentSpace(**values)
    types = _payload_types()
    if kind not in types:
        raise ValueError(f"Unknown run-data codec: {kind!r}.")
    return types[kind](**values)


def dumps(value: object) -> bytes:
    """Pack JSON metadata and raw tensor blocks into one versioned binary blob.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: bytes
    """
    blocks = []

    def pack(item: object) -> object:
        """Pack.

        :param item: Item.
        :type item: object
        :return: Result of this operation.
        :rtype: object
        """
        if isinstance(item, bytes):
            index = len(blocks)
            blocks.append(item)
            return {"__blob__": index}
        if isinstance(item, dict):
            return {key: pack(value) for key, value in item.items()}
        if isinstance(item, list):
            return [pack(value) for value in item]
        return item

    tree = pack(encode(value))
    header = json.dumps(
        {"tree": tree, "lengths": [len(block) for block in blocks]},
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return b"".join((b"PEPCOMPASS1", len(header).to_bytes(8, "big"), header, *blocks))


def loads(value: bytes) -> object:
    """Restore a value serialized with :func:`dumps`.

    :param value: Value consumed by this conversion or transformation.
    :type value: bytes
    :return: Result of this operation.
    :rtype: object
    """
    magic = b"PEPCOMPASS1"
    header_start = len(magic) + 8
    if not value.startswith(magic) or len(value) < header_start:
        raise ValueError("Unsupported run payload format.")
    offset = header_start + int.from_bytes(value[len(magic) : header_start], "big")
    header = json.loads(value[header_start:offset])
    blocks = []
    for length in header["lengths"]:
        if not isinstance(length, int) or length < 0 or offset + length > len(value):
            raise ValueError("Truncated run payload.")
        blocks.append(value[offset : offset + length])
        offset += length
    if offset != len(value):
        raise ValueError("Unexpected trailing run payload bytes.")

    def unpack(item: object) -> object:
        """Unpack.

        :param item: Item.
        :type item: object
        :return: Result of this operation.
        :rtype: object
        """
        if isinstance(item, dict):
            if set(item) == {"__blob__"}:
                return blocks[item["__blob__"]]
            return {key: unpack(value) for key, value in item.items()}
        if isinstance(item, list):
            return [unpack(value) for value in item]
        return item

    return decode(unpack(header["tree"]))
