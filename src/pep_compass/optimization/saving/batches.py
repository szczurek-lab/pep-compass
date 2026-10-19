"""Ordered batch identities, compressed provenance and shared field payloads."""

from __future__ import annotations

import weakref
from collections.abc import Sequence

import torch

from pep_compass.data_schemas.candidates import (
    CandidateBatch,
    ObjectField,
    OptionalField,
    TensorField,
)
from pep_compass.optimization.saving.backends import SQLiteStorageBackend
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

_SEQUENCE_FIELDS = {"mutation.parent_sequence", "local_enumeration.center_sequence"}


class BatchStore:
    """Retain structural provenance independently of full numerical batch states.

    :param backend: Canonical run database.
    :type backend: SQLiteStorageBackend
    :param lineage: Whether to retain structural parent/child relations.
    :type lineage: bool
    """

    def __init__(self, backend: SQLiteStorageBackend, lineage: bool) -> None:
        """Initialize this object with the declared configuration.

        :param backend: Backend.
        :type backend: SQLiteStorageBackend
        :param lineage: Candidate-aligned provenance identities.
        :type lineage: bool
        :rtype: None
        """
        self.backend = backend
        self.lineage = lineage
        self._batches: dict[int, tuple[weakref.ReferenceType[CandidateBatch], int]] = {}
        self._locations: dict[int, tuple[int, int]] = {}

    def peptide_id(self, sequence: str) -> int:
        """Intern a retained sequence exactly once.

        :param sequence: Evaluated or explicitly reconstructed peptide string.
        :type sequence: str
        :return: Database-local sequence identity.
        :rtype: int
        """
        self.backend.execute(
            "INSERT OR IGNORE INTO peptides(sequence) VALUES (?)", (sequence,)
        )
        return self.backend.execute(
            "SELECT peptide_id FROM peptides WHERE sequence=?", (sequence,)
        ).fetchone()[0]

    def bind_candidate(self, candidate_id: int, peptide_id: int) -> None:
        """Associate an occurrence with its immutable canonical sequence.

        :param candidate_id: Run-local candidate occurrence identity.
        :type candidate_id: int
        :param peptide_id: Existing canonical peptide identity.
        :type peptide_id: int
        :rtype: None
        :raises ValueError: If the occurrence already refers to another sequence.
        """
        self.backend.execute(
            "INSERT OR IGNORE INTO candidate_instances(candidate_id) VALUES (?)",
            (candidate_id,),
        )
        existing = self.backend.execute(
            "SELECT peptide_id FROM candidate_instances WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()[0]
        if existing is not None and existing != peptide_id:
            raise ValueError("A candidate identity cannot change its sequence.")
        self.backend.execute(
            "UPDATE candidate_instances SET peptide_id=? WHERE candidate_id=?",
            (peptide_id, candidate_id),
        )

    def register(
        self,
        batch: CandidateBatch,
        *,
        execution_id: int | None,
        iteration_id: int | None = None,
        parent_batch_id: int | None = None,
        full: bool = False,
    ) -> int | None:
        """Record ordered members and optionally their complete field state.

        :param batch: Immutable candidate batch; tensor fields must remain unchanged.
        :type batch: CandidateBatch
        :param execution_id: Producer invocation or None for the initial input.
        :type execution_id: int | None
        :param iteration_id: Enclosing logical iteration.
        :type iteration_id: int | None
        :param parent_batch_id: Immediate transformation input, when known.
        :type parent_batch_id: int | None
        :param full: Also retain sequences, latent origins and batch fields.
        :type full: bool
        :return: Structural batch identity, or None when no retention is needed.
        :rtype: int | None
        :raises ValueError: If a retained batch lacks valid candidate identities.
        """
        if not self.lineage and not full:
            return None
        known = self._batches.get(id(batch))
        logger.debug(
            "Retaining batch candidates=%s full=%s lineage=%s existing=%s.",
            len(batch),
            full,
            self.lineage,
            known is not None and known[0]() is batch,
        )
        if known is not None and known[0]() is batch:
            batch_id = known[1]
        else:
            field = batch.fields.get("lineage.candidate_id")
            if not isinstance(field, TensorField):
                raise ValueError("Retained batches require lineage.candidate_id.")
            candidate_ids = [
                int(value) for value in field.values.detach().cpu().tolist()
            ]
            cursor = self.backend.execute(
                "INSERT INTO batches(producer_execution_id,candidate_count) VALUES (?,?)",
                (execution_id, len(batch)),
            )
            batch_id = cursor.lastrowid
            for position, candidate_id in enumerate(candidate_ids):
                self.backend.execute(
                    "INSERT OR IGNORE INTO candidate_instances(candidate_id) VALUES (?)",
                    (candidate_id,),
                )
                self.backend.execute(
                    "INSERT INTO batch_members VALUES (?,?,?)",
                    (batch_id, position, candidate_id),
                )
            if self.lineage and execution_id is not None:
                self._record_lineage(
                    batch_id,
                    batch,
                    candidate_ids,
                    parent_batch_id,
                    execution_id,
                    iteration_id,
                )
            for position, candidate_id in enumerate(candidate_ids):
                self._locations[candidate_id] = (batch_id, position)
            key = id(batch)
            self._batches[key] = (
                weakref.ref(
                    batch,
                    lambda reference, identity=key: self._batches.pop(identity, None),
                ),
                batch_id,
            )
        batch._bind_lineage_source(batch_id)
        if full:
            self._retain_fields(batch_id, batch)
        return batch_id

    def _record_lineage(
        self,
        batch_id: int,
        batch: CandidateBatch,
        candidate_ids: Sequence[int],
        parent_batch_id: int | None,
        execution_id: int,
        iteration_id: int | None,
    ) -> None:
        """Record lineage.

        :param batch_id: Retained ordered batch identity.
        :type batch_id: int
        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param candidate_ids: Candidate ids.
        :type candidate_ids: Sequence[int]
        :param parent_batch_id: Immediate source batch identity, when available.
        :type parent_batch_id: int | None
        :param execution_id: Run-local step invocation identity.
        :type execution_id: int
        :param iteration_id: Run-local logical iteration identity.
        :type iteration_id: int | None
        :rtype: None
        """
        immediate = {}
        if parent_batch_id is not None:
            immediate = {
                row[1]: (parent_batch_id, row[0])
                for row in self.backend.execute(
                    "SELECT position,candidate_id FROM batch_members WHERE batch_id=? ORDER BY position",
                    (parent_batch_id,),
                )
            }
        parent_field = batch.fields.get("lineage.parent_candidate_id")
        parents = (
            parent_field.values.detach().cpu().tolist()
            if isinstance(parent_field, TensorField)
            else [-1] * len(batch)
        )
        edges = []
        ordered_input = (
            self.backend.execute(
                "SELECT candidate_id FROM batch_members WHERE batch_id=? ORDER BY position",
                (parent_batch_id,),
            ).fetchall()
            if parent_batch_id is not None
            else []
        )
        for position, (candidate_id, parent_id) in enumerate(
            zip(candidate_ids, parents, strict=True)
        ):
            source = (
                batch._row_origins[position] if batch._row_origins is not None else None
            )
            if source is not None:
                edges.append((position, source[0], source[1]))
                continue
            if len(ordered_input) == len(batch) and ordered_input[position][0] in {
                candidate_id,
                int(parent_id),
            }:
                edges.append((position, parent_batch_id, position))
                continue
            parent = (
                immediate.get(candidate_id)
                or immediate.get(int(parent_id))
                or self._locations.get(int(parent_id))
                or self._locations.get(candidate_id)
            )
            if parent is not None:
                edges.append((position, parent[0], parent[1]))
        # Compress only unambiguous equal-stride mappings or singleton-parent broadcasts.
        index = 0
        while index < len(edges):
            start, parent_batch, parent_start = edges[index]
            end = index + 1
            mode = "pairwise"
            if end < len(edges) and edges[end] == (
                start + 1,
                parent_batch,
                parent_start,
            ):
                mode = "broadcast"
            while end < len(edges):
                expected_parent = (
                    parent_start if mode == "broadcast" else parent_start + end - index
                )
                if edges[end] != (start + end - index, parent_batch, expected_parent):
                    break
                end += 1
            length = end - index
            self.backend.execute(
                "INSERT INTO batch_lineage(child_batch_id,child_start,child_end,parent_batch_id,parent_start,parent_end,mapping_kind,execution_id,iteration_id) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    batch_id,
                    start,
                    start + length,
                    parent_batch,
                    parent_start,
                    parent_start + (1 if mode == "broadcast" else length),
                    mode,
                    execution_id,
                    iteration_id,
                ),
            )
            index = end

    def _retain_fields(self, batch_id: int, batch: CandidateBatch) -> None:
        """Retain fields.

        :param batch_id: Retained ordered batch identity.
        :type batch_id: int
        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :rtype: None
        """
        members = self.backend.execute(
            "SELECT candidate_id FROM batch_members WHERE batch_id=? ORDER BY position",
            (batch_id,),
        ).fetchall()
        for row, sequence in zip(members, batch.sequences, strict=True):
            self.bind_candidate(row[0], self.peptide_id(sequence))
        values = {
            "latent_origins": batch.latent_origins,
            **{
                name: value
                for name, value in batch.fields.items()
                if name != "lineage.candidate_id"
            },
        }
        for name, value in values.items():
            if name in _SEQUENCE_FIELDS:
                optional = isinstance(value, OptionalField)
                strings = value.values if optional else value
                if isinstance(strings, ObjectField):
                    value = {
                        "peptide_ids": [
                            self.peptide_id(sequence) if sequence is not None else None
                            for sequence in strings.values
                        ],
                        "valid": value.valid if optional else None,
                    }
            payload_id = self.backend.payload(value)
            existing = self.backend.execute(
                "SELECT payload_id FROM batch_payloads WHERE batch_id=? AND field_name=?",
                (batch_id, name),
            ).fetchone()
            if existing is not None and existing[0] != payload_id:
                raise ValueError("Retained batch fields were mutated in place.")
            self.backend.execute(
                "INSERT OR IGNORE INTO batch_payloads VALUES (?,?,?)",
                (batch_id, name, payload_id),
            )

    def load(self, batch_id: int) -> CandidateBatch:
        """Reassemble a full batch using its canonical sequence and member rows.

        :param batch_id: Retained batch identity.
        :type batch_id: int
        :return: Aligned batch with original field dtypes and order.
        :rtype: CandidateBatch
        :raises LookupError: If the complete batch state was not retained.
        """
        members = self.backend.execute(
            "SELECT m.candidate_id,p.sequence FROM batch_members m JOIN candidate_instances c USING(candidate_id) LEFT JOIN peptides p USING(peptide_id) WHERE m.batch_id=? ORDER BY m.position",
            (batch_id,),
        ).fetchall()
        values = {
            row[0]: self.backend.load_payload(row[1])
            for row in self.backend.execute(
                "SELECT field_name,payload_id FROM batch_payloads WHERE batch_id=?",
                (batch_id,),
            )
        }
        latent = values.pop("latent_origins", None)
        if latent is None or any(row[1] is None for row in members):
            raise LookupError("Complete batch fields were not retained.")
        values["lineage.candidate_id"] = TensorField(
            torch.tensor(
                [row[0] for row in members], dtype=torch.long, device=latent.device
            )
        )  # (B,)
        for name in _SEQUENCE_FIELDS & values.keys():
            stored = values[name]
            if isinstance(stored, dict) and "peptide_ids" in stored:
                values[name] = ObjectField(
                    [
                        self.backend.execute(
                            "SELECT sequence FROM peptides WHERE peptide_id=?",
                            (identity,),
                        ).fetchone()[0]
                        if identity is not None
                        else None
                        for identity in stored["peptide_ids"]
                    ]
                )
                if stored.get("valid") is not None:
                    values[name] = OptionalField(values[name], stored["valid"])
        result = CandidateBatch([row[1] for row in members], latent, values)
        result._bind_lineage_source(batch_id)
        return result
