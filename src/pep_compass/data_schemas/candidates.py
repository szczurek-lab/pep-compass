"""Columnar candidate data passed between optimization steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import torch


def _as_index_tensor(
    indices: torch.Tensor | Sequence[int], device: torch.device
) -> torch.Tensor:
    """Normalize candidate indices for a tensor operation.

    :param indices: Zero-based row indices; ordering and repeated occurrences are preserved.
    :type indices: torch.Tensor | Sequence[int]
    :param device: Computation device accepted by the numerical backend.
    :type device: torch.device
    :return: Result of this operation.
    :rtype: torch.Tensor
    """
    return torch.as_tensor(indices, dtype=torch.long, device=device)


class BatchField(ABC):
    """A column or shared value attached to a candidate batch."""

    @abstractmethod
    def validate_length(self, batch_size: int) -> None:
        """Validate compatibility with a candidate batch.

        :param batch_size: Number of candidates in the owning batch.
        :type batch_size: int
        :raises ValueError: If the field is not aligned with the batch.

        :rtype: None
        """

    @abstractmethod
    def select(self, indices: torch.Tensor | Sequence[int]) -> "BatchField":
        """Select rows aligned with candidate indices.

        :param indices: Zero-based row indices; ordering and repeated occurrences are preserved.
        :type indices: torch.Tensor | Sequence[int]
        :return: Result of this operation.
        :rtype: BatchField
        """

    @abstractmethod
    def concatenate(self, other: "BatchField") -> "BatchField":
        """Concatenate two compatible fields.

        :param other: Other.
        :type other: BatchField
        :return: Result of this operation.
        :rtype: BatchField
        """

    @abstractmethod
    def optional(self, batch_size: int, *, valid: bool) -> "OptionalField":
        """Represent the field with an explicit validity mask.

        :param batch_size: Batch size.
        :type batch_size: int
        :param valid: Valid.
        :type valid: bool
        :return: Result of this operation.
        :rtype: OptionalField
        """


@dataclass(frozen=True)
class TensorField(BatchField):
    """Candidate-aligned tensor values."""

    values: torch.Tensor

    def validate_length(self, batch_size: int) -> None:
        """Validate length.

        :param batch_size: Batch size.
        :type batch_size: int
        :rtype: None
        """
        if self.values.ndim == 0 or self.values.shape[0] != batch_size:
            raise ValueError(
                "Tensor field must have batch size as its first dimension."
            )

    def select(self, indices: torch.Tensor | Sequence[int]) -> "TensorField":
        """Select.

        :param indices: Zero-based row indices; ordering and repeated occurrences are preserved.
        :type indices: torch.Tensor | Sequence[int]
        :return: Result of this operation.
        :rtype: TensorField
        """
        normalized = _as_index_tensor(indices, self.values.device)
        return TensorField(self.values.index_select(0, normalized))

    def concatenate(self, other: BatchField) -> "TensorField":
        """Concatenate.

        :param other: Other.
        :type other: BatchField
        :return: Result of this operation.
        :rtype: TensorField
        """
        if not isinstance(other, TensorField):
            raise TypeError(
                "Tensor fields can only be concatenated with tensor fields."
            )
        if self.values.device != other.values.device:
            raise ValueError("Concatenated tensor fields must use the same device.")
        return TensorField(torch.cat((self.values, other.values), dim=0))

    def optional(self, batch_size: int, *, valid: bool) -> "OptionalField":
        """Optional.

        :param batch_size: Batch size.
        :type batch_size: int
        :param valid: Valid.
        :type valid: bool
        :return: Result of this operation.
        :rtype: OptionalField
        """
        self.validate_length(batch_size)
        mask = torch.full(
            (batch_size,), valid, dtype=torch.bool, device=self.values.device
        )
        return OptionalField(self, mask)


@dataclass(frozen=True)
class ObjectField(BatchField):
    """Candidate-aligned non-tensor values."""

    values: tuple[Any, ...]

    def __init__(self, values: Sequence[Any]) -> None:
        """Initialize this object with the declared configuration.

        :param values: Values consumed by this conversion or transformation.
        :type values: Sequence[Any]
        :rtype: None
        """
        object.__setattr__(self, "values", tuple(values))

    def validate_length(self, batch_size: int) -> None:
        """Validate length.

        :param batch_size: Batch size.
        :type batch_size: int
        :rtype: None
        """
        if len(self.values) != batch_size:
            raise ValueError("Object field length must equal the candidate batch size.")

    def select(self, indices: torch.Tensor | Sequence[int]) -> "ObjectField":
        """Select.

        :param indices: Zero-based row indices; ordering and repeated occurrences are preserved.
        :type indices: torch.Tensor | Sequence[int]
        :return: Result of this operation.
        :rtype: ObjectField
        """
        rows = (
            indices.detach().cpu().tolist()
            if isinstance(indices, torch.Tensor)
            else list(indices)
        )
        return ObjectField([self.values[index] for index in rows])

    def concatenate(self, other: BatchField) -> "ObjectField":
        """Concatenate.

        :param other: Other.
        :type other: BatchField
        :return: Result of this operation.
        :rtype: ObjectField
        """
        if not isinstance(other, ObjectField):
            raise TypeError(
                "Object fields can only be concatenated with object fields."
            )
        return ObjectField((*self.values, *other.values))

    def optional(self, batch_size: int, *, valid: bool) -> "OptionalField":
        """Optional.

        :param batch_size: Batch size.
        :type batch_size: int
        :param valid: Valid.
        :type valid: bool
        :return: Result of this operation.
        :rtype: OptionalField
        """
        self.validate_length(batch_size)
        return OptionalField(self, torch.full((batch_size,), valid, dtype=torch.bool))


@dataclass(frozen=True)
class SharedField(BatchField):
    """One immutable value shared by all candidates in a batch."""

    value: Any

    def validate_length(self, batch_size: int) -> None:
        """Validate length.

        :param batch_size: Batch size.
        :type batch_size: int
        :rtype: None
        """
        if batch_size < 0:
            raise ValueError("Batch size cannot be negative.")

    def select(self, indices: torch.Tensor | Sequence[int]) -> "SharedField":
        """Select.

        :param indices: Zero-based row indices; ordering and repeated occurrences are preserved.
        :type indices: torch.Tensor | Sequence[int]
        :return: Result of this operation.
        :rtype: SharedField
        """
        return self

    def concatenate(self, other: BatchField) -> "SharedField":
        """Concatenate.

        :param other: Other.
        :type other: BatchField
        :return: Result of this operation.
        :rtype: SharedField
        """
        if not isinstance(other, SharedField) or other.value != self.value:
            raise ValueError(
                "Shared fields can only be merged when their values are equal."
            )
        return self

    def optional(self, batch_size: int, *, valid: bool) -> "OptionalField":
        """Optional.

        :param batch_size: Batch size.
        :type batch_size: int
        :param valid: Valid.
        :type valid: bool
        :return: Result of this operation.
        :rtype: OptionalField
        """
        mask = torch.full((batch_size,), valid, dtype=torch.bool)
        return OptionalField(ObjectField([self.value] * batch_size), mask)


@dataclass(frozen=True)
class OptionalField(BatchField):
    """Candidate-aligned values accompanied by a validity mask."""

    values: BatchField
    valid: torch.Tensor

    def validate_length(self, batch_size: int) -> None:
        """Validate length.

        :param batch_size: Batch size.
        :type batch_size: int
        :rtype: None
        """
        self.values.validate_length(batch_size)
        if self.valid.dtype != torch.bool:
            raise ValueError("Optional field masks must have boolean dtype.")
        if self.valid.ndim != 1 or self.valid.shape[0] != batch_size:
            raise ValueError("Optional field mask must align with the candidate batch.")

    def select(self, indices: torch.Tensor | Sequence[int]) -> "OptionalField":
        """Select.

        :param indices: Zero-based row indices; ordering and repeated occurrences are preserved.
        :type indices: torch.Tensor | Sequence[int]
        :return: Result of this operation.
        :rtype: OptionalField
        """
        normalized = _as_index_tensor(indices, self.valid.device)
        return OptionalField(
            self.values.select(indices),
            self.valid.index_select(0, normalized),
        )

    def concatenate(self, other: BatchField) -> "OptionalField":
        """Concatenate.

        :param other: Other.
        :type other: BatchField
        :return: Result of this operation.
        :rtype: OptionalField
        """
        if not isinstance(other, OptionalField):
            raise TypeError(
                "Optional fields can only be concatenated with optional fields."
            )
        if self.valid.device != other.valid.device:
            raise ValueError("Optional field masks must use the same device.")
        return OptionalField(
            self.values.concatenate(other.values),
            torch.cat((self.valid, other.valid), dim=0),
        )

    def optional(self, batch_size: int, *, valid: bool) -> "OptionalField":
        """Optional.

        :param batch_size: Batch size.
        :type batch_size: int
        :param valid: Valid.
        :type valid: bool
        :return: Result of this operation.
        :rtype: OptionalField
        """
        self.validate_length(batch_size)
        if valid:
            return self
        return OptionalField(self.values, torch.zeros_like(self.valid))


@dataclass(frozen=True)
class Candidate:
    """One sequence paired with the latent position that generated it.

    ``latent_origin`` is deliberately not the result of re-encoding
    ``sequence``. It preserves the concrete trajectory position from which the
    sequence was produced.

    :param sequence: Peptide sequence.
    :type sequence: str
    :param latent_origin: Generating latent-space position.
    :type latent_origin: torch.Tensor
    """

    sequence: str
    latent_origin: torch.Tensor


@dataclass(frozen=True)
class CandidateBatch:
    """Immutable, columnar collection passed between optimization steps."""

    sequences: tuple[str, ...]
    latent_origins: torch.Tensor
    fields: Mapping[str, BatchField] = field(default_factory=dict)
    _row_origins: tuple[tuple[int, int] | None, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __init__(
        self,
        sequences: Sequence[str],
        latent_origins: torch.Tensor,
        fields: Mapping[str, BatchField] | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param sequences: Peptide sequences in input order.
        :type sequences: Sequence[str]
        :param latent_origins: Generating latent coordinates aligned with candidate rows.
        :type latent_origins: torch.Tensor
        :param fields: Fields.
        :type fields: Mapping[str, BatchField] | None
        :rtype: None
        """
        normalized_sequences = tuple(sequences)
        if latent_origins.ndim < 2:
            raise ValueError("Latent origins must have shape [batch, ...].")
        if len(normalized_sequences) != latent_origins.shape[0]:
            raise ValueError("Sequences and latent origins must have equal batch size.")
        normalized_fields = dict(fields or {})
        for value in normalized_fields.values():
            value.validate_length(len(normalized_sequences))
        object.__setattr__(self, "sequences", normalized_sequences)
        object.__setattr__(self, "latent_origins", latent_origins)
        object.__setattr__(self, "fields", MappingProxyType(normalized_fields))
        object.__setattr__(self, "_row_origins", None)

    def _bind_lineage_source(self, identity: int) -> None:
        """Bind observer-owned row references without changing computational data.

        REMARK: This private provenance cache is excluded from equality and codecs.
        Binding is performed after an observer records the immutable batch state.

        :param identity: Identity.
        :type identity: int
        :rtype: None
        """
        object.__setattr__(
            self, "_row_origins", tuple((identity, index) for index in range(len(self)))
        )

    def _inherit_lineage(
        self, result: CandidateBatch, rows: Sequence[int] | None = None
    ) -> CandidateBatch:
        """Copy aligned source-row references across an immutable transformation.

        :param result: Result.
        :type result: CandidateBatch
        :param rows: Zero-based source row indices, or None to preserve the complete order.
        :type rows: Sequence[int] | None
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        origins = self._row_origins
        if origins is not None and rows is not None:
            origins = tuple(origins[index] for index in rows)
        object.__setattr__(result, "_row_origins", origins)
        return result

    def __reduce__(self) -> tuple[object, tuple[object, object, dict[str, object]]]:
        """Serialize the field mapping through the validated public constructor.

        :return: Result of this operation.
        :rtype: tuple[object, tuple[object, object, dict[str, object]]]
        """
        return type(self), (self.sequences, self.latent_origins, dict(self.fields))

    def __len__(self) -> int:
        """Return the number of contained elements.

        :return: Result of this operation.
        :rtype: int
        """
        return len(self.sequences)

    def __iter__(self) -> Iterator[Candidate]:
        """Iter.

        :return: Result of this operation.
        :rtype: Iterator[Candidate]
        """
        for index, sequence in enumerate(self.sequences):
            yield Candidate(sequence, self.latent_origins[index])

    def select(self, indices: torch.Tensor | Sequence[int]) -> "CandidateBatch":
        """Select candidates and every aligned optimization field.

        :param indices: Zero-based row indices; ordering and repeated occurrences are preserved.
        :type indices: torch.Tensor | Sequence[int]
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        normalized = _as_index_tensor(indices, self.latent_origins.device)
        rows = normalized.detach().cpu().tolist()
        return self._inherit_lineage(
            CandidateBatch(
                sequences=[self.sequences[index] for index in rows],
                latent_origins=self.latent_origins.index_select(0, normalized),
                fields={
                    name: value.select(indices) for name, value in self.fields.items()
                },
            ),
            rows,
        )

    def repeat_from_parents(
        self, parent_indices: torch.Tensor | Sequence[int]
    ) -> "CandidateBatch":
        """Expand candidates according to one-to-many parent indices.

        :param parent_indices: Parent indices.
        :type parent_indices: torch.Tensor | Sequence[int]
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        return self.select(parent_indices)

    def with_sequences(self, sequences: Sequence[str]) -> "CandidateBatch":
        """Replace sequences and invalidate sequence-dependent oracle fields.

        :param sequences: Replacement sequences aligned with existing rows.
        :return: Batch retaining latent provenance and sequence-independent fields.

        REMARK: Scores belong to evaluated sequences. Copying parent scores to
        new mutations would incorrectly label unobserved candidates as evaluated.

        :type sequences: Sequence[str]
        :rtype: CandidateBatch
        """
        sequences = tuple(sequences)
        fields = (
            self.fields
            if sequences == self.sequences
            else {
                name: value
                for name, value in self.fields.items()
                if not name.startswith("oracle.")
            }
        )
        return self._inherit_lineage(
            CandidateBatch(sequences, self.latent_origins, fields)
        )

    def with_field(self, name: str, value: BatchField) -> "CandidateBatch":
        """Return a batch with one added or replaced optimization field.

        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :param value: Value consumed by this conversion or transformation.
        :type value: BatchField
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        value.validate_length(len(self))
        updated = dict(self.fields)
        updated[name] = value
        return self._inherit_lineage(
            CandidateBatch(self.sequences, self.latent_origins, updated)
        )

    def without_fields(self, names: Sequence[str]) -> "CandidateBatch":
        """Return a batch without selected fields while sharing retained data.

        :param names: Exact field names to remove.
        :return: Batch preserving sequences, latent origins, and other fields.

        :type names: Sequence[str]
        :rtype: CandidateBatch
        """
        removed = frozenset(names)
        return self._inherit_lineage(
            CandidateBatch(
                self.sequences,
                self.latent_origins,
                {
                    name: value
                    for name, value in self.fields.items()
                    if name not in removed
                },
            )
        )

    @classmethod
    def concatenate(cls, batches: Sequence["CandidateBatch"]) -> "CandidateBatch":
        """Concatenate branches while preserving fields absent from some branches.

        :param batches: Batches.
        :type batches: Sequence['CandidateBatch']
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        if not batches:
            raise ValueError(
                "At least one candidate batch is required for concatenation."
            )
        device = batches[0].latent_origins.device
        if any(batch.latent_origins.device != device for batch in batches):
            raise ValueError(
                "All concatenated latent batches must use the same device."
            )

        all_names = set().union(*(batch.fields.keys() for batch in batches))
        merged_fields: dict[str, BatchField] = {}
        for name in sorted(all_names):
            present = [batch.fields[name] for batch in batches if name in batch.fields]
            missing = len(present) != len(batches)
            normalized: list[BatchField] = []
            prototype = present[0]
            for batch in batches:
                if name in batch.fields:
                    field_value = batch.fields[name]
                    normalized.append(
                        field_value.optional(len(batch), valid=True)
                        if missing
                        else field_value
                    )
                else:
                    normalized.append(_missing_like(prototype, len(batch)))
            merged = normalized[0]
            for value in normalized[1:]:
                merged = merged.concatenate(value)
            merged_fields[name] = merged

        result = cls(
            sequences=[sequence for batch in batches for sequence in batch.sequences],
            latent_origins=torch.cat(
                [batch.latent_origins for batch in batches], dim=0
            ),
            fields=merged_fields,
        )

        object.__setattr__(
            result,
            "_row_origins",
            tuple(
                origin
                for batch in batches
                for origin in (batch._row_origins or (None,) * len(batch))
            ),
        )
        return result


def _missing_like(prototype: BatchField, batch_size: int) -> OptionalField:
    """Create invalid placeholder rows compatible with an existing field.

    :param prototype: Prototype.
    :type prototype: BatchField
    :param batch_size: Batch size.
    :type batch_size: int
    :return: Result of this operation.
    :rtype: OptionalField
    """
    if isinstance(prototype, OptionalField):
        prototype = prototype.values
    if isinstance(prototype, TensorField):
        shape = (batch_size, *prototype.values.shape[1:])
        empty = torch.zeros(
            shape, dtype=prototype.values.dtype, device=prototype.values.device
        )
        return TensorField(empty).optional(batch_size, valid=False)
    if isinstance(prototype, ObjectField):
        return ObjectField([None] * batch_size).optional(batch_size, valid=False)
    if isinstance(prototype, SharedField):
        return ObjectField([None] * batch_size).optional(batch_size, valid=False)
    raise TypeError("Unsupported batch field type for optional branch merging.")
