"""CSV-backed consumer of lifecycle events emitted by ``Step.__call__``.

``RuntimeRunner._build_tracker`` selects this implementation when a run has an
output directory. The engine sees only the ``StepTracker`` contract and has no
knowledge of CSV files or retention settings.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from pep_compass.optimization.tracking import ExecutionScope, StepTracker, TrackingLevel

if TYPE_CHECKING:
    from pep_compass.data.optimization import CandidateBatch
    from pep_compass.optimization.engine.execution.context import OptimizationContext
    from pep_compass.optimization.engine.execution.step import Step


CandidateSnapshots = Literal["none", "oracle", "all"]


@dataclass(frozen=True, slots=True)
class _TrackingHandle:
    """Internal lifecycle state for one tracked step execution."""

    execution_id: int
    started_at: float
    oracle_calls: int
    generated_candidates: int
    input_count: int
    checkpoint_file: str | None = None


class CSVStepTracker(StepTracker):
    """Stream depth-aware optimization records to normalized CSV files.

    ``short`` writes oracle summaries. ``normal`` and ``all`` write scalar
    summaries for every enabled step. ``candidate_snapshots`` independently
    controls candidate-row retention; checkpoints required for replay are
    always streamed for local enumeration and SORBES points in normal/all.
    ``store_latents`` and
    ``store_fields`` control columns within candidate rows; ``field_names``
    restricts serialized fields further.

    Rows are serialized and flushed immediately. The tracker does not retain
    completed ``CandidateBatch`` objects, so Python and PyTorch may release
    their storage once no operation references them. Closing is guaranteed by
    ``PepCompassPipeline.run`` in a ``finally`` block.
    """

    def __init__(
        self,
        output_directory: str | Path,
        *,
        level: TrackingLevel = "normal",
        max_depth: int | None = None,
        store_latents: bool = False,
        store_fields: bool = False,
        field_names: list[str] | tuple[str, ...] | None = None,
        candidate_snapshots: CandidateSnapshots = "none",
        run_id: str | None = None,
        variant_id: str | None = None,
    ) -> None:
        if level not in {"short", "normal", "all"}:
            raise ValueError("Tracking level must be short, normal, or all.")
        if candidate_snapshots not in {"none", "oracle", "all"}:
            raise ValueError("Candidate snapshots must be none, oracle, or all.")
        self.level = level
        self.max_depth = max_depth
        self.store_latents = store_latents
        self.store_fields = store_fields
        self.field_names = frozenset(field_names) if field_names is not None else None
        self.candidate_snapshots = candidate_snapshots
        self.run_id = run_id
        self.variant_id = variant_id
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._next_execution_id = 0
        self._step_stream = (self.output_directory / "steps.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._candidate_stream = (self.output_directory / "candidates.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._trajectory_stream = (self.output_directory / "trajectory_points.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._local_enumeration_stream = (
            self.output_directory / "local_enumerations.csv"
        ).open("w", newline="", encoding="utf-8")
        self._checkpoint_directory = self.output_directory / "checkpoints"
        self._trajectory_checkpoint_directory = self._checkpoint_directory / "trajectory"
        self._local_checkpoint_directory = self._checkpoint_directory / "local_enumeration"
        self._trajectory_checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self._local_checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self._next_trajectory_latent_index = 0
        self._step_writer = csv.DictWriter(
            self._step_stream,
            fieldnames=(
                "execution_id",
                "run_id",
                "variant_id",
                "step_name",
                "path",
                "depth",
                "loop_indices",
                "branch_names",
                "branch_indices",
                "input_size",
                "output_size",
                "status",
                "duration_seconds",
                "oracle_calls_before",
                "oracle_calls_after",
                "generated_candidates_before",
                "generated_candidates_after",
                "error",
            ),
        )
        self._candidate_writer = csv.DictWriter(
            self._candidate_stream,
            fieldnames=(
                "execution_id",
                "candidate_index",
                "sequence",
                "latent_origin",
                "fields",
            ),
        )
        self._trajectory_writer = csv.DictWriter(
            self._trajectory_stream,
            fieldnames=(
                "execution_id",
                "run_id",
                "variant_id",
                "loop_indices",
                "trajectory_id",
                "trajectory_index",
                "rng_stream_seed",
                "trajectory_step",
                "point_id",
                "sequence",
                "latent_index",
                "checkpoint_file",
                "checkpoint_index",
                "candidate_id",
                "parent_candidate_id",
                "adjusted_time_step",
            ),
        )
        self._local_enumeration_writer = csv.DictWriter(
            self._local_enumeration_stream,
            fieldnames=(
                "execution_id",
                "run_id",
                "variant_id",
                "loop_indices",
                "input_count",
                "checkpoint_file",
                "output_count",
                "output_sequences_sha256",
            ),
        )
        self._step_writer.writeheader()
        self._candidate_writer.writeheader()
        self._trajectory_writer.writeheader()
        self._local_enumeration_writer.writeheader()

    def begin_step(
        self,
        step: "Step",
        batch: "CandidateBatch",
        scope: ExecutionScope,
        context: "OptimizationContext",
    ) -> _TrackingHandle:
        with self._lock:
            execution_id = self._next_execution_id
            self._next_execution_id += 1
        checkpoint_file = None
        if self.level in {"normal", "all"} and step.name == "LocalEnumeration":
            checkpoint_file = self._write_local_enumeration_checkpoint(
                execution_id, batch, context
            )
        return _TrackingHandle(
            execution_id=execution_id,
            started_at=perf_counter(),
            oracle_calls=context.state.oracle_calls,
            generated_candidates=context.state.generated_candidates,
            input_count=len(batch),
            checkpoint_file=checkpoint_file,
        )

    def end_step(
        self,
        handle: Any,
        step: "Step",
        input_batch: "CandidateBatch",
        output_batch: "CandidateBatch",
        scope: ExecutionScope,
        context: "OptimizationContext",
    ) -> None:
        from pep_compass.optimization.components.oracles.base import Oracle

        is_oracle = isinstance(step, Oracle)
        if self.level == "short" and not is_oracle:
            return
        step_row = {
            "execution_id": handle.execution_id,
            "run_id": self.run_id,
            "variant_id": self.variant_id,
            "step_name": step.name,
            "path": "/".join(scope.path),
            "depth": scope.depth,
            "loop_indices": json.dumps(scope.loop_indices),
            "branch_names": json.dumps(scope.branch_names),
            "branch_indices": json.dumps(scope.branch_indices),
            "input_size": len(input_batch),
            "output_size": len(output_batch),
            "status": "completed",
            "duration_seconds": perf_counter() - handle.started_at,
            "oracle_calls_before": handle.oracle_calls,
            "oracle_calls_after": context.state.oracle_calls,
            "generated_candidates_before": handle.generated_candidates,
            "generated_candidates_after": context.state.generated_candidates,
            "error": "",
        }
        candidate_rows = []
        if self.candidate_snapshots == "all" or (
            self.candidate_snapshots == "oracle" and is_oracle
        ):
            candidate_rows = self._candidate_rows(handle.execution_id, output_batch)
        with self._lock:
            trajectory_rows = (
                self._trajectory_rows(handle.execution_id, output_batch, scope)
                if self.level in {"normal", "all"}
                and step.name == "SorbesWalker"
                else []
            )
            local_enumeration_row = (
                self._local_enumeration_row(
                    handle, output_batch, scope
                )
                if self.level in {"normal", "all"}
                and step.name == "LocalEnumeration"
                else None
            )
            self._step_writer.writerow(step_row)
            self._candidate_writer.writerows(candidate_rows)
            self._trajectory_writer.writerows(trajectory_rows)
            if local_enumeration_row is not None:
                self._local_enumeration_writer.writerow(local_enumeration_row)
            self._step_stream.flush()
            self._candidate_stream.flush()
            self._trajectory_stream.flush()
            self._local_enumeration_stream.flush()

    def fail_step(self, handle, step, input_batch, scope, context, error):
        row = {
            "execution_id": handle.execution_id,
            "run_id": self.run_id,
            "variant_id": self.variant_id,
            "step_name": step.name,
            "path": "/".join(scope.path),
            "depth": scope.depth,
            "loop_indices": json.dumps(scope.loop_indices),
            "branch_names": json.dumps(scope.branch_names),
            "branch_indices": json.dumps(scope.branch_indices),
            "input_size": len(input_batch),
            "output_size": 0,
            "status": "failed",
            "duration_seconds": perf_counter() - handle.started_at,
            "oracle_calls_before": handle.oracle_calls,
            "oracle_calls_after": context.state.oracle_calls,
            "generated_candidates_before": handle.generated_candidates,
            "generated_candidates_after": context.state.generated_candidates,
            "error": f"{type(error).__name__}: {error}",
        }
        with self._lock:
            self._step_writer.writerow(row)
            self._step_stream.flush()

    def _candidate_rows(
        self,
        execution_id: int,
        batch: "CandidateBatch",
    ) -> list[dict[str, Any]]:
        rows = []
        for index, sequence in enumerate(batch.sequences):
            latent_origin = ""
            if self.store_latents:
                latent_origin = json.dumps(
                    batch.latent_origins[index].detach().cpu().tolist(),
                    separators=(",", ":"),
                )
            rows.append(
                {
                    "execution_id": execution_id,
                    "candidate_index": index,
                    "sequence": sequence,
                    "latent_origin": latent_origin,
                    "fields": (
                        self._serialize_fields(batch, index)
                        if self.store_fields
                        else ""
                    ),
                }
            )
        return rows

    def _serialize_fields(self, batch: "CandidateBatch", index: int) -> str:
        from pep_compass.data.optimization import (
            ObjectField,
            OptionalField,
            SharedField,
            TensorField,
        )

        values: dict[str, Any] = {}
        for name, field_value in batch.fields.items():
            if self.field_names is not None and name not in self.field_names:
                continue
            if isinstance(field_value, TensorField):
                values[name] = field_value.values[index].detach().cpu().tolist()
            elif isinstance(field_value, ObjectField):
                values[name] = repr(field_value.values[index])
            elif isinstance(field_value, SharedField):
                values[name] = repr(field_value.value)
            elif isinstance(field_value, OptionalField):
                values[name] = {"valid": bool(field_value.valid[index].item())}
        return json.dumps(values, separators=(",", ":"))

    def _trajectory_rows(
        self,
        execution_id: int,
        batch: "CandidateBatch",
        scope: ExecutionScope,
    ) -> list[dict[str, Any]]:
        """Store compact SORBES checkpoints required for deterministic replay."""
        from pep_compass.data.optimization import ObjectField, TensorField

        object_names = ("tracking.trajectory_id", "point_id")
        tensor_names = (
            "tracking.trajectory_index",
            "tracking.rng_stream_seed",
            "tracking.trajectory_step",
            "walker.adjusted_time_step",
        )
        objects = {name: batch.fields.get(name) for name in object_names}
        tensors = {name: batch.fields.get(name) for name in tensor_names}
        if not isinstance(objects["tracking.trajectory_id"], ObjectField):
            return []
        checkpoint_path = (
            self._trajectory_checkpoint_directory / f"{execution_id:08d}.pt"
        )
        torch.save(
            batch.latent_origins.detach().to(device="cpu", dtype=torch.float32),
            checkpoint_path,
        )
        latent_start = self._next_trajectory_latent_index
        self._next_trajectory_latent_index += len(batch)
        candidate_ids = batch.fields.get("lineage.candidate_id")
        parent_ids = batch.fields.get("lineage.parent_candidate_id")
        rows = []
        for index, sequence in enumerate(batch.sequences):
            rows.append(
                {
                    "execution_id": execution_id,
                    "run_id": self.run_id,
                    "variant_id": self.variant_id,
                    "loop_indices": json.dumps(scope.loop_indices),
                    "trajectory_id": objects["tracking.trajectory_id"].values[index],
                    "trajectory_index": self._tensor_scalar(
                        tensors["tracking.trajectory_index"], index
                    ),
                    "rng_stream_seed": self._tensor_scalar(
                        tensors["tracking.rng_stream_seed"], index
                    ),
                    "trajectory_step": self._tensor_scalar(
                        tensors["tracking.trajectory_step"], index
                    ),
                    "point_id": (
                        objects["point_id"].values[index]
                        if isinstance(objects["point_id"], ObjectField)
                        else ""
                    ),
                    "sequence": sequence,
                    "latent_index": latent_start + index,
                    "checkpoint_file": str(
                        checkpoint_path.relative_to(self.output_directory)
                    ),
                    "checkpoint_index": index,
                    "candidate_id": self._tensor_scalar(candidate_ids, index),
                    "parent_candidate_id": self._tensor_scalar(parent_ids, index),
                    "adjusted_time_step": self._tensor_scalar(
                        tensors["walker.adjusted_time_step"], index
                    ),
                }
            )
        return rows

    def _local_enumeration_row(
        self,
        handle: _TrackingHandle,
        output_batch: "CandidateBatch",
        scope: ExecutionScope,
    ) -> dict[str, Any]:
        """Write one local-enumeration result digest linked to its input shard."""
        payload = json.dumps(
            list(output_batch.sequences),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "execution_id": handle.execution_id,
            "run_id": self.run_id,
            "variant_id": self.variant_id,
            "loop_indices": json.dumps(scope.loop_indices),
            "input_count": handle.input_count,
            "checkpoint_file": handle.checkpoint_file or "",
            "output_count": len(output_batch),
            "output_sequences_sha256": hashlib.sha256(payload).hexdigest(),
        }

    def _write_local_enumeration_checkpoint(
        self,
        execution_id: int,
        batch: "CandidateBatch",
        context: "OptimizationContext",
    ) -> str:
        """Persist an isolated replay boundary before local enumeration.

        :param execution_id: Lifecycle execution identity of LocalEnumeration.
        :param batch: Input candidates at the boundary.
        :param context: Current deterministic execution context.
        :return: Path relative to the tracking directory parent.

        REMARK: Saving the RNG state is essential for resuming from this boundary;
        the global run seed alone only reproduces execution from the beginning.
        """
        from pep_compass.data.optimization import TensorField

        candidate_ids = batch.fields.get("lineage.candidate_id")
        parent_ids = batch.fields.get("lineage.parent_candidate_id")
        path = self._local_checkpoint_directory / f"{execution_id:08d}.pt"
        payload: dict[str, Any] = {
            "sequences": tuple(batch.sequences),
            "latent_origins": batch.latent_origins.detach().to(
                device="cpu", dtype=torch.float32
            ),
            "candidate_ids": (
                candidate_ids.values.detach().to(device="cpu", dtype=torch.long)
                if isinstance(candidate_ids, TensorField)
                else torch.full((len(batch),), -1, dtype=torch.long)
            ),
            "parent_candidate_ids": (
                parent_ids.values.detach().to(device="cpu", dtype=torch.long)
                if isinstance(parent_ids, TensorField)
                else torch.full((len(batch),), -1, dtype=torch.long)
            ),
            "seed": context.seed,
            "numpy_rng_state": context.rng.bit_generator.state,
            "numpy_global_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["torch_cuda_rng_states"] = torch.cuda.get_rng_state_all()
        torch.save(payload, path)
        return str(path.relative_to(self.output_directory))

    @staticmethod
    def _tensor_scalar(field: Any, index: int) -> int | float | str:
        """Return one scalar tensor field value or an empty CSV value."""
        from pep_compass.data.optimization import TensorField

        if not isinstance(field, TensorField):
            return ""
        return field.values[index].item()

    def close(self) -> None:
        with self._lock:
            self._step_stream.flush()
            self._candidate_stream.flush()
            self._trajectory_stream.flush()
            self._local_enumeration_stream.flush()
            self._step_stream.close()
            self._candidate_stream.close()
            self._trajectory_stream.close()
            self._local_enumeration_stream.close()
