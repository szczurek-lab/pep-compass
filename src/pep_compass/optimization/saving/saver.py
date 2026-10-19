"""Record canonical evaluations, provenance, measurements and replay boundaries."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch

from pep_compass.data_schemas.candidates import CandidateBatch, TensorField
from pep_compass.data_schemas.execution import StorageSettings
from pep_compass.optimization.saving.backends import SQLiteStorageBackend
from pep_compass.optimization.saving.batches import BatchStore
from pep_compass.optimization.saving.metrics import Measurement, MemorySampler
from pep_compass.optimization.tracking import ExecutionScope, StepTracker
from pep_compass.utils.logger import get_custom_logger

if TYPE_CHECKING:
    from pep_compass.optimization.engine.execution.context import OptimizationContext
    from pep_compass.optimization.engine.execution.state import OptimizationState
    from pep_compass.steps.base import Step


logger = get_custom_logger(__name__)


def sequence_digest(sequences: Sequence[str]) -> str:
    """Hash ordered sequences using the canonical replay encoding.

    :param sequences: Peptide sequences in input order.
    :type sequences: Sequence[str]
    :return: Result of this operation.
    :rtype: str
    """
    raw = json.dumps(list(sequences), ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def rng_snapshot(context: OptimizationContext) -> dict[str, object]:
    """Capture supported random streams without consuming their values.

    :param context: Scoped execution services, shared run state and random generator.
    :type context: OptimizationContext
    :return: Result of this operation.
    :rtype: dict[str, object]
    """
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_context": context.rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_initialized()
        else None,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


@dataclass
class ExecutionHandle:
    """One active invocation; owns no input or output batch objects."""

    identity: int
    input_batch_id: int | None
    measurement: Measurement | None
    checkpoint_id: int | None = None


@dataclass
class IterationHandle:
    """One active logical iteration and its enclosing invocation."""

    identity: int
    owner_execution_id: int
    measurement: Measurement | None
    started: float
    key: tuple[object, ...]


def _measurements(measurement: Measurement | None) -> tuple[float | int | None, ...]:
    """Measurements.

    :param measurement: Measurement.
    :type measurement: Measurement | None
    :return: Result of this operation.
    :rtype: tuple[float | int | None, ...]
    """
    if measurement is None:
        return (None, None, None, None)
    return (
        measurement.duration_seconds,
        measurement.memory_start_bytes,
        measurement.memory_peak_bytes,
        measurement.memory_end_bytes,
    )


class Saver(StepTracker):
    """Observe the step lifecycle without owning algorithmic batch selection.

    :param backend: Shared relational storage implementation.
    :type backend: SQLiteStorageBackend
    :param settings: Validated retention and measurement settings.
    :type settings: StorageSettings
    """

    def __init__(
        self, backend: SQLiteStorageBackend, settings: StorageSettings
    ) -> None:
        """Initialize this object with the declared configuration.

        :param backend: Backend.
        :type backend: SQLiteStorageBackend
        :param settings: Validated settings or a resolved settings manifest.
        :type settings: StorageSettings
        :rtype: None
        """
        self.backend = backend
        self.settings = settings
        self.batches = BatchStore(backend, settings.save_lineage)
        self.sampler = MemorySampler(settings.metrics_level != "none")
        self.started_at = perf_counter()
        self._next_execution = 0
        self._stack: list[ExecutionHandle] = []
        self._iteration_stack: list[IterationHandle] = []
        self._last_iteration_time: dict[tuple[object, ...], float] = {}
        self._iterations = 0
        self._evaluations = 0
        self._events = 0
        self.closed = False

    @property
    def iteration_id(self) -> int | None:
        """Return the innermost active logical iteration identity.

        :return: Result of this operation.
        :rtype: int | None
        """
        return self._iteration_stack[-1].identity if self._iteration_stack else None

    def _execution_id(self) -> int:
        """Execution id.

        :return: Result of this operation.
        :rtype: int
        """
        identity = self._next_execution
        self._next_execution += 1
        return identity

    def begin_step(
        self,
        step: Step,
        batch: CandidateBatch,
        scope: ExecutionScope,
        context: OptimizationContext,
    ) -> ExecutionHandle:
        """Register invocation identity before preparation and transformation.

        :param step: Step.
        :type step: Step
        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param scope: Nested step, iteration and branch coordinates.
        :type scope: ExecutionScope
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: ExecutionHandle
        """
        from pep_compass.steps.flow.strategies.local_enumeration import LocalEnumeration

        identity = self._execution_id()
        self.backend.execute(
            "INSERT INTO step_executions(execution_id,node_id,parent_execution_id,iteration_id,scope_json,status,phase) VALUES (?,?,?,?,?,?,?)",
            (
                identity,
                context.node_id or "root",
                self._stack[-1].identity if self._stack else None,
                self.iteration_id,
                json.dumps(scope.__dict__),
                "running",
                "prepare_call",
            ),
        )
        full = self.settings.reconstruction == "trajectories"
        input_id = self.batches.register(
            batch,
            execution_id=self._stack[-1].identity if self._stack else None,
            iteration_id=self.iteration_id,
            full=full,
        )
        handle = ExecutionHandle(identity, input_id, self.sampler.begin())
        self._stack.append(handle)
        self.backend.execute(
            "UPDATE step_executions SET input_batch_id=?,input_count=? WHERE execution_id=?",
            (input_id, len(batch) if self.sampler.enabled else None, identity),
        )
        try:
            if self.settings.reconstruction != "none" and isinstance(
                step, LocalEnumeration
            ):
                input_id = self.batches.register(batch, execution_id=None, full=True)
                handle.input_batch_id = input_id
                self.backend.execute(
                    "UPDATE step_executions SET input_batch_id=? WHERE execution_id=?",
                    (input_id, identity),
                )
                state = {
                    item.name: getattr(context.state, item.name)
                    for item in fields(context.state)
                    if item.name
                    not in {"observer", "observations", "objective_directions"}
                }
                payload = self.backend.payload(
                    {
                        "batch_id": input_id,
                        "evaluation_count": self._evaluations,
                        "state": state,
                        "rng": rng_snapshot(context),
                        "seed": context.seed,
                        "scope": scope,
                        "step_state": context.step_state,
                    }
                )
                handle.checkpoint_id = self.backend.execute(
                    "INSERT INTO checkpoints(node_id,execution_id,iteration_id,scope_json,phase,payload_id) VALUES (?,?,?,?,?,?)",
                    (
                        context.node_id,
                        identity,
                        self.iteration_id,
                        json.dumps(scope.__dict__),
                        "before_prepare_call",
                        payload,
                    ),
                ).lastrowid
        except BaseException as error:
            self.fail_step(handle, step, batch, scope, context, error)
            raise
        return handle

    def execution_started(self, handle: ExecutionHandle) -> None:
        """Mark the transformation phase after successful call preparation.

        :param handle: Handle.
        :type handle: ExecutionHandle
        :rtype: None
        """
        self.backend.execute(
            "UPDATE step_executions SET phase='execute' WHERE execution_id=?",
            (handle.identity,),
        )

    def end_step(
        self,
        handle: ExecutionHandle,
        step: Step,
        input_batch: CandidateBatch,
        output_batch: CandidateBatch,
        scope: ExecutionScope,
        context: OptimizationContext,
    ) -> None:
        """Complete identity, optional measurements and retained batch references.

        :param handle: Handle.
        :type handle: ExecutionHandle
        :param step: Step.
        :type step: Step
        :param input_batch: Aligned batch before transformation.
        :type input_batch: CandidateBatch
        :param output_batch: Aligned batch after transformation.
        :type output_batch: CandidateBatch
        :param scope: Nested step, iteration and branch coordinates.
        :type scope: ExecutionScope
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :rtype: None
        """
        output_id = self.batches.register(
            output_batch,
            execution_id=handle.identity,
            iteration_id=self.iteration_id,
            parent_batch_id=handle.input_batch_id,
            full=self.settings.reconstruction == "trajectories",
        )
        measurement = self.sampler.end(handle.measurement)
        self.backend.execute(
            "UPDATE step_executions SET output_batch_id=?,output_count=?,duration_seconds=?,memory_start_bytes=?,memory_peak_bytes=?,memory_end_bytes=?,status='completed' WHERE execution_id=?",
            (
                output_id,
                len(output_batch) if self.sampler.enabled else None,
                *_measurements(measurement),
                handle.identity,
            ),
        )
        if handle.checkpoint_id is not None:
            self.backend.execute(
                "UPDATE checkpoints SET output_count=?,output_sha256=? WHERE checkpoint_id=?",
                (
                    len(output_batch),
                    sequence_digest(output_batch.sequences),
                    handle.checkpoint_id,
                ),
            )
        if self.settings.reconstruction == "trajectories" and step.category == "walker":
            self.capture(
                output_batch,
                name="trajectory_point",
                context=context,
                kind="trajectory",
            )
        if self.settings.component_data != "none":
            value = step.display_info()
            if self.settings.component_data == "full":
                value = {
                    "display": value,
                    "field_payloads": {
                        name: self.backend.payload(field)
                        for name, field in output_batch.fields.items()
                        if name
                        not in {
                            "mutation.parent_sequence",
                            "local_enumeration.center_sequence",
                        }
                    },
                }
            self.backend.execute(
                "INSERT INTO component_records(node_id,execution_id,name,payload_id) VALUES (?,?,?,?)",
                (
                    context.node_id,
                    handle.identity,
                    step.name,
                    self.backend.payload(value),
                ),
            )
        self._finish_handle(handle)
        self._flush_event()

    def begin_precompute(
        self, step: Step, context: OptimizationContext
    ) -> ExecutionHandle:
        """Open static preparation so its internal stages have an invocation identity.

        :param step: Step.
        :type step: Step
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: ExecutionHandle
        """
        identity = self._execution_id()
        self.backend.execute(
            "INSERT INTO step_executions(execution_id,node_id,parent_execution_id,scope_json,status,phase) VALUES (?,?,?,?,?,?)",
            (
                identity,
                context.node_id or "root",
                self._stack[-1].identity if self._stack else None,
                json.dumps(context.scope.__dict__),
                "running",
                "precompute",
            ),
        )
        handle = ExecutionHandle(identity, None, self.sampler.begin())
        self._stack.append(handle)
        return handle

    def end_precompute(
        self, handle: ExecutionHandle, error: BaseException | None
    ) -> None:
        """Close static preparation without manufacturing candidate batch data.

        :param handle: Handle.
        :type handle: ExecutionHandle
        :param error: Original failure, or None when execution succeeded.
        :type error: BaseException | None
        :rtype: None
        """
        self.backend.execute(
            "UPDATE step_executions SET duration_seconds=?,memory_start_bytes=?,memory_peak_bytes=?,memory_end_bytes=?,status=?,error=? WHERE execution_id=?",
            (
                *_measurements(self.sampler.end(handle.measurement)),
                "failed" if error is not None else "completed",
                None if error is None else f"{type(error).__name__}: {error}",
                handle.identity,
            ),
        )
        self._finish_handle(handle)

    def fail_step(
        self,
        handle: ExecutionHandle,
        step: Step,
        input_batch: CandidateBatch,
        scope: ExecutionScope,
        context: OptimizationContext,
        error: BaseException,
    ) -> None:
        """Preserve the original failure and finalize active measurement scopes.

        :param handle: Handle.
        :type handle: ExecutionHandle
        :param step: Step.
        :type step: Step
        :param input_batch: Aligned batch before transformation.
        :type input_batch: CandidateBatch
        :param scope: Nested step, iteration and branch coordinates.
        :type scope: ExecutionScope
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :param error: Original failure, or None when execution succeeded.
        :type error: BaseException
        :rtype: None
        """
        self.backend.execute(
            "UPDATE step_executions SET duration_seconds=?,memory_start_bytes=?,memory_peak_bytes=?,memory_end_bytes=?,status='failed',error=? WHERE execution_id=?",
            (
                *_measurements(self.sampler.end(handle.measurement)),
                f"{type(error).__name__}: {error}",
                handle.identity,
            ),
        )
        while (
            self._iteration_stack
            and self._iteration_stack[-1].owner_execution_id == handle.identity
        ):
            iteration = self._iteration_stack.pop()
            self.backend.execute(
                "UPDATE iterations SET status='failed',duration_seconds=?,memory_start_bytes=?,memory_peak_bytes=?,memory_end_bytes=? WHERE iteration_id=?",
                (
                    *_measurements(self.sampler.end(iteration.measurement)),
                    iteration.identity,
                ),
            )
        self._finish_handle(handle)

    def _finish_handle(self, handle: ExecutionHandle) -> None:
        """Finish handle.

        :param handle: Handle.
        :type handle: ExecutionHandle
        :rtype: None
        """
        if self._stack and self._stack[-1] is handle:
            self._stack.pop()
        elif handle in self._stack:
            raise RuntimeError("Unbalanced step lifecycle events.")

    def _flush_event(self) -> None:
        """Flush event.

        :rtype: None
        """
        self._events += 1
        if self._events % self.settings.flush_every == 0:
            self.backend.flush()

    def link_evaluated_candidates(self, batch: CandidateBatch, objective: str) -> None:
        """Resolve scored occurrences without recording another evaluation.

        :param batch: Candidate rows whose objective values are already known.
        :type batch: CandidateBatch
        :param objective: Configured objective identity used by the run cache.
        :type objective: str
        :rtype: None
        :raises ValueError: If an occurrence lacks provenance or a recorded score.
        """
        if not self.settings.save_lineage and self.settings.reconstruction == "none":
            return
        identities = batch.fields.get("lineage.candidate_id")
        if not isinstance(identities, TensorField):
            raise ValueError("Scored occurrences require lineage.candidate_id.")
        for sequence, candidate_id in zip(
            batch.sequences, identities.values.detach().cpu().tolist(), strict=True
        ):
            row = self.backend.execute(
                "SELECT p.peptide_id FROM peptides p JOIN evaluations e ON e.peptide_id=p.peptide_id WHERE p.sequence=? AND e.objective=? LIMIT 1",
                (sequence, objective),
            ).fetchone()
            if row is None:
                raise ValueError(
                    "A reused objective value requires a recorded evaluation."
                )
            self.batches.bind_candidate(int(candidate_id), row[0])

    def observations(
        self,
        objective: str,
        sequences: Sequence[str],
        scores: Sequence[float],
        *,
        direction: str,
        lineage: dict[str, list[int]] | None,
        configuration_sha256: str | None = None,
    ) -> None:
        """Append every actual evaluation; later selection cannot delete it.

        :param objective: Objective field identity shared by scoring and optimization.
        :type objective: str
        :param sequences: Peptide sequences in input order.
        :type sequences: Sequence[str]
        :param scores: Scalar scores or a score tensor aligned with the corresponding candidates.
        :type scores: Sequence[float]
        :param direction: Objective ordering, minimize or maximize.
        :type direction: str
        :param lineage: Candidate-aligned provenance identities.
        :type lineage: dict[str, list[int]] | None
        :param configuration_sha256: Digest identifying the configured predictor within this run.
        :type configuration_sha256: str | None
        :rtype: None
        """
        if not self._stack:
            raise RuntimeError("Evaluations require an active step invocation.")
        self.backend.execute(
            "INSERT OR IGNORE INTO objectives VALUES (?,?,?)",
            (objective, direction, configuration_sha256),
        )
        existing = self.backend.execute(
            "SELECT direction,configuration_sha256 FROM objectives WHERE objective=?",
            (objective,),
        ).fetchone()
        if existing[0] != direction or existing[1] != configuration_sha256:
            raise ValueError(
                "A named objective cannot change its configured identity within a run."
            )
        for index, (sequence, score) in enumerate(zip(sequences, scores, strict=True)):
            peptide_id = self.batches.peptide_id(sequence)
            ids = (lineage or {}).get("lineage.candidate_id")
            candidate_id = (
                int(ids[index])
                if ids is not None
                and (
                    self.settings.save_lineage or self.settings.reconstruction != "none"
                )
                else None
            )
            if candidate_id is not None:
                self.batches.bind_candidate(candidate_id, peptide_id)
            self.backend.execute(
                "INSERT INTO evaluations(objective,peptide_id,candidate_id,execution_id,iteration_id,evaluation_index,score) VALUES (?,?,?,?,?,?,?)",
                (
                    objective,
                    peptide_id,
                    candidate_id,
                    self._stack[-1].identity,
                    self.iteration_id,
                    self._evaluations,
                    float(score),
                ),
            )
            self._evaluations += 1

    def capture(
        self,
        batch: CandidateBatch,
        *,
        name: str,
        context: OptimizationContext | None = None,
        kind: str = "explicit",
    ) -> None:
        """Label a retained batch reference without duplicating its contents.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext | None
        :param kind: Kind of recorded artifact or exception type, as declared by this operation.
        :type kind: str
        :rtype: None
        """
        if self.settings.reconstruction == "none":
            raise ValueError(
                "Snapshots require reconstruction=boundaries or trajectories."
            )
        execution_id = self._stack[-1].identity if self._stack else None
        batch_id = self.batches.register(
            batch, execution_id=execution_id, iteration_id=self.iteration_id, full=True
        )
        self.backend.execute(
            "INSERT INTO candidate_sets(name,batch_id,node_id,execution_id,iteration_id,kind) VALUES (?,?,?,?,?,?)",
            (
                name,
                batch_id,
                context.node_id if context is not None else None,
                execution_id,
                self.iteration_id,
                kind,
            ),
        )

    def iteration_started(self, context: OptimizationContext, index: int) -> None:
        """Open a nested iteration before any of its child invocations.

        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :param index: Index.
        :type index: int
        :rtype: None
        """
        key = (context.node_id, context.scope.loop_indices, context.scope.branch_names)
        identity = self.backend.execute(
            "INSERT INTO iterations(loop_execution_id,parent_iteration_id,node_id,iteration_index,scope_json,status) VALUES (?,?,?,?,?,?)",
            (
                self._stack[-1].identity,
                self.iteration_id,
                context.node_id or "root",
                index,
                json.dumps(context.scope.__dict__),
                "running",
            ),
        ).lastrowid
        self._iteration_stack.append(
            IterationHandle(
                identity,
                self._stack[-1].identity,
                self.sampler.begin(),
                perf_counter(),
                key,
            )
        )

    def iteration_completed(
        self, context: OptimizationContext, index: int, trajectories: int
    ) -> None:
        """Aggregate child leaf executions once, excluding composite durations.

        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :param index: Index.
        :type index: int
        :param trajectories: Independent trajectories per input seed.
        :type trajectories: int
        :rtype: None
        """
        handle = self._iteration_stack.pop()
        now = perf_counter()
        previous = self._last_iteration_time.get(handle.key)
        self._last_iteration_time[handle.key] = now
        # Nested iteration membership is resolved recursively; leaf rows avoid double counting.
        sum_row = self.backend.execute(
            """WITH RECURSIVE descendants(iteration_id) AS (
            SELECT ? UNION ALL SELECT i.iteration_id FROM iterations i JOIN descendants d ON i.parent_iteration_id=d.iteration_id
        ) SELECT SUM(e.duration_seconds) FROM step_executions e
        WHERE e.iteration_id IN (SELECT iteration_id FROM descendants)
        AND NOT EXISTS(SELECT 1 FROM step_executions child WHERE child.parent_execution_id=e.execution_id)""",
            (handle.identity,),
        ).fetchone()
        endpoints = self.backend.execute(
            """SELECT input_count,output_count FROM step_executions WHERE iteration_id=?
            AND (parent_execution_id=? OR parent_execution_id IS NULL) ORDER BY execution_id""",
            (handle.identity, handle.owner_execution_id),
        ).fetchall()
        measurement = self.sampler.end(handle.measurement)
        measured = self.sampler.enabled
        self.backend.execute(
            """UPDATE iterations SET trajectories=?,elapsed_seconds=?,duration_seconds=?,memory_start_bytes=?,memory_peak_bytes=?,memory_end_bytes=?,interval_seconds=?,step_duration_sum_seconds=?,input_count=?,output_count=?,status='completed' WHERE iteration_id=?""",
            (
                trajectories if measured else None,
                now - self.started_at if measured else None,
                *_measurements(measurement),
                now - previous if previous is not None and measured else None,
                sum_row[0] if measured else None,
                endpoints[0][0] if endpoints else None,
                endpoints[-1][1] if endpoints else None,
                handle.identity,
            ),
        )
        self._iterations += 1
        self._flush_event()

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        input_count: int | None = None,
        output_count: int | None = None,
    ) -> Iterator[None]:
        """Measure one internal strategy stage only at the requested detail level.

        :param name: Stable stage name within the enclosing component invocation.
        :type name: str
        :param input_count: Number of input rows, if meaningful for this stage.
        :type input_count: int | None
        :param output_count: Number of output rows, if known before execution.
        :type output_count: int | None
        :return: Context manager recording aggregated stage measurements.
        :rtype: Iterator[None]
        """
        if self.settings.metrics_level != "basic_and_components":
            yield
            return
        measurement = self.sampler.begin()
        status = "completed"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            self.backend.execute(
                """INSERT INTO execution_metrics(execution_id,iteration_id,stage_name,duration_seconds,memory_start_bytes,memory_peak_bytes,memory_end_bytes,input_count,output_count,status) VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(execution_id,stage_name) DO UPDATE SET duration_seconds=execution_metrics.duration_seconds+excluded.duration_seconds,memory_peak_bytes=MAX(execution_metrics.memory_peak_bytes,excluded.memory_peak_bytes),memory_end_bytes=excluded.memory_end_bytes,input_count=execution_metrics.input_count+excluded.input_count,output_count=execution_metrics.output_count+excluded.output_count,status=CASE WHEN excluded.status='failed' THEN excluded.status ELSE execution_metrics.status END""",
                (
                    self._stack[-1].identity,
                    self.iteration_id,
                    name,
                    *_measurements(self.sampler.end(measurement)),
                    input_count,
                    output_count,
                    status,
                ),
            )

    def finalize(
        self,
        result: CandidateBatch | None,
        state: OptimizationState,
        *,
        status: str = "completed",
        error: str | None = None,
    ) -> dict[str, object]:
        """Commit aggregate status; evaluated data already resides in its sole table.

        :param result: Result.
        :type result: CandidateBatch | None
        :param state: State.
        :type state: OptimizationState
        :param status: Lifecycle status written to the run database.
        :type status: str
        :param error: Original failure, or None when execution succeeded.
        :type error: str | None
        :return: Result of this operation.
        :rtype: dict[str, object]
        """
        summary = {
            "duration_seconds": perf_counter() - self.started_at,
            "oracle_calls": state.oracle_calls,
            "generated_candidates": state.generated_candidates,
            "completed_iterations": state.completed_iterations,
            "iteration_events": self._iterations,
            "objective_name": state.last_objective,
            "terminal_count": len(result) if result is not None else None,
            "error": error,
        }
        self.backend.execute(
            "UPDATE runs SET status=?,summary_json=?",
            (status, json.dumps(summary, allow_nan=False)),
        )
        self.close()
        return summary

    def close(self) -> None:
        """Stop sampling and commit; the lazy reader owns the database lifetime.

        :rtype: None
        """
        if not self.closed:
            self.sampler.close()
            self.backend.flush()
            self.closed = True
