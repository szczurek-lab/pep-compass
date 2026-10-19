"""Lifecycle contract for every executable pipeline node.

``PipelineBuilder`` creates a tree whose composite and leaf nodes all inherit
from :class:`Step`. ``ExecutablePipeline.run`` invokes its root. The public
``__call__`` method supplies iteration preparation, tracking, resource
sampling, timing, and failure reporting before delegating only the actual
transformation to :meth:`_execute`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from time import perf_counter
from typing import TYPE_CHECKING

from pep_compass.data_schemas.candidates import CandidateBatch
from pep_compass.data_schemas.contracts import StepContract
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.utils.configuration import Configurable
from pep_compass.utils.logger import get_custom_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pep_compass.utils.configuration import ConfigValue


logger = get_custom_logger(__name__)


class Step(Configurable, ABC):
    """Transform a candidate batch within the common execution lifecycle.

    Subclasses normally implement only
    * :meth:`_execute`.
    Override
    * :meth:`precompute` for state independent of the current batch,
    * :meth:`prepare_call` for state refreshed immediately before each call.

    Subclasses **must not** override :meth:`__call__`, because
    doing so would bypass tracking, memory sampling, timing, and error events.

    Composite subclasses are
    * ``Flow``,
    * ``Loop``,
    * ``Parallel``
    under ``steps.flow.strategies``. Leaf implementations live under their
    category-specific ``strategies`` packages.
    """

    category = "custom"
    kind = "step"
    input_fields = ()
    output_fields = ()
    optional_inputs = ()
    invalidated_prefixes = ()
    display_parameters = ()
    required_resources = ()

    def graph_children(
        self,
    ) -> dict[str, Step | tuple[Step, ...] | dict[str, tuple[Step, ...]]]:
        """Return constructor-owned child steps; ordinary components have none.

        :return: Result of this operation.
        :rtype: dict[str, Step | tuple[Step, ...] | dict[str, tuple[Step, ...]]]
        """
        return {}

    def to_config(self) -> dict[str, ConfigValue]:
        """Serialize constructor-owned graph children exactly once per traversal.

        :return: Result of this operation.
        :rtype: dict[str, ConfigValue]
        """
        from pep_compass.utils.configuration import plain

        return {**super().to_config(), **plain(self.graph_children())}

    def contract(self) -> StepContract:
        """Resolve data-flow metadata for this configured object.

        :return: Result of this operation.
        :rtype: StepContract
        """
        values = self.configuration

        def declared(
            value: Sequence[str] | Callable[[Mapping[str, object]], Sequence[str]],
        ) -> tuple[str, ...]:
            """Declared.

            :param value: Value consumed by this conversion or transformation.
            :type value: Sequence[str] | Callable[[Mapping[str, object]], Sequence[str]]
            :return: Result of this operation.
            :rtype: tuple[str, ...]
            """
            return tuple(value(values) if callable(value) else value)

        return StepContract(
            declared(self.input_fields),
            declared(self.optional_inputs),
            declared(self.output_fields),
            tuple(self.invalidated_prefixes),
            tuple(self.required_resources),
        )

    def candidate_bound(self, count: int) -> int | None:
        """Return an upper output count, or None for a declared unknown bound.

        :param count: Count.
        :type count: int
        :return: Result of this operation.
        :rtype: int | None
        """
        bound = getattr(self, "output_bound", None)
        return bound(self.configuration, count) if bound is not None else count

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Keep lifecycle orchestration central, including preparation failures.

        :param kwargs: Additional named arguments forwarded to the underlying interface.
        :type kwargs: object
        :rtype: None
        """
        super().__init_subclass__(**kwargs)
        if "__call__" in cls.__dict__:
            raise TypeError("Executable steps implement _execute, not __call__.")
        preparation = cls.__dict__.get("precompute")
        if preparation is not None:

            @wraps(preparation)
            def tracked_precompute(self, context: OptimizationContext) -> None:
                """Tracked precompute.

                :param context: Scoped execution services, shared run state and random generator.
                :type context: OptimizationContext
                :rtype: None
                """
                scoped = context.enter_node(getattr(self, "node_id", context.node_id))
                handle = (
                    scoped.saver.begin_precompute(self, scoped)
                    if scoped.saver is not None
                    else None
                )
                error = None
                try:
                    return preparation(self, scoped)
                except BaseException as caught:
                    error = caught
                    raise
                finally:
                    if scoped.saver is not None:
                        scoped.saver.end_precompute(handle, error)

            cls.precompute = tracked_precompute

    @property
    def name(self) -> str:
        """Return the stable tracking name of this step.

        :return: Result of this operation.
        :rtype: str
        """
        return getattr(self, "display_name", None) or self.__class__.__name__

    def display_info(self) -> dict:
        """Return configured parameters selected for compact graph presentation.

        Components may override this method; it must not perform computation.

        :return: Small mapping of meaningful parameter names and values.

        :rtype: dict
        """
        configuration = getattr(self, "configuration", {})
        selected = self.display_parameters
        return {key: configuration[key] for key in selected if key in configuration}

    def precompute(self, context: OptimizationContext) -> None:
        """Prepare reusable state once before the optimization run.

        :param context: Runtime services shared by all steps.
        :type context: OptimizationContext

        :rtype: None
        """

    def prepare_call(self, batch: CandidateBatch, context: OptimizationContext) -> None:
        """Refresh batch-dependent state immediately before every transformation.

        :param batch: Incoming aligned candidate batch.
        :param context: Scoped services and fresh run state.

        :type batch: CandidateBatch
        :type context: OptimizationContext
        :rtype: None
        """

    def __call__(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Apply preparation, transformation and observation as one lifecycle.

        :param batch: Aligned input candidates; implementations do not mutate it.
        :param context: Scoped run services and state.
        :return: Aligned output batch.
        :raises Exception: Original preparation, computation or recording error.

        :type batch: CandidateBatch
        :type context: OptimizationContext
        :rtype: CandidateBatch
        """
        # Invocation scope and before-step observations
        step_context = context.enter_step(self.name).enter_node(
            getattr(self, "node_id", context.node_id)
        )
        step_context.stability_monitor.sample(
            f"step.before:{'/'.join(step_context.scope.path)}",
            batch,
            step_context.state,
        )
        enabled = step_context.tracker.is_enabled(step_context.scope)
        handle = None
        if enabled:
            handle = step_context.tracker.begin_step(
                self, batch, step_context.scope, step_context
            )
        started_at = perf_counter()

        # Preparation, transformation and completion share failure reporting
        try:
            self.prepare_call(batch, step_context)
            if step_context.saver is not None:
                step_context.saver.execution_started(handle)
            result = self._execute(batch, step_context)
            if not isinstance(result, CandidateBatch):
                raise TypeError(
                    f"{self.name} must return CandidateBatch, got {type(result).__name__}."
                )
            if step_context.saver is not None:
                result = self._ensure_lineage(batch, result, step_context)
            if enabled:
                step_context.tracker.end_step(
                    handle, self, batch, result, step_context.scope, step_context
                )
            step_context.stability_monitor.sample(
                f"step.after:{'/'.join(step_context.scope.path)}",
                result,
                step_context.state,
            )
        except BaseException as error:
            if enabled:
                try:
                    step_context.tracker.fail_step(
                        handle, self, batch, step_context.scope, step_context, error
                    )
                except Exception:
                    logger.error(
                        "Failed to record the original step error path=%s.",
                        step_context.scope.path,
                        exc_info=True,
                    )
            raise

        # Compact diagnostics never retain candidate batches
        logger.debug(
            "Step path=%s input_candidates=%s output_candidates=%s duration_seconds=%.6f.",
            "/".join(step_context.scope.path),
            len(batch),
            len(result),
            perf_counter() - started_at,
        )
        return result

    @staticmethod
    def _ensure_lineage(
        batch: CandidateBatch, result: CandidateBatch, context: OptimizationContext
    ) -> CandidateBatch:
        """Assign fresh identities when a custom transformation changes a sequence.

        Existing component-owned fresh identities are preserved. Filters retaining
        a sequence keep its identity, including repeated occurrences in a batch.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param result: Result.
        :type result: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        import torch

        from pep_compass.data_schemas.candidates import TensorField

        source = batch.fields.get("lineage.candidate_id")
        output = result.fields.get("lineage.candidate_id")
        origins = (
            dict(zip(source.values.detach().cpu().tolist(), batch.sequences))
            if isinstance(source, TensorField)
            else {}
        )
        ids = (
            output.values.detach().cpu().tolist()
            if isinstance(output, TensorField)
            else [None] * len(result)
        )
        parents_field = result.fields.get("lineage.parent_candidate_id")
        parents = (
            parents_field.values.detach().cpu().tolist()
            if isinstance(parents_field, TensorField)
            else [-1] * len(result)
        )
        changed = False
        for index, (identity, sequence) in enumerate(
            zip(ids, result.sequences, strict=True)
        ):
            if identity is None or (
                identity in origins and origins[identity] != sequence
            ):
                parents[index] = identity if identity is not None else -1
                ids[index] = context.state.next_candidate_ids(1)[0]
                changed = True
        if changed:
            device = result.latent_origins.device
            result = result.with_field(
                "lineage.candidate_id",
                TensorField(torch.tensor(ids, dtype=torch.long, device=device)),
            )  # (B,)
            result = result.with_field(
                "lineage.parent_candidate_id",
                TensorField(torch.tensor(parents, dtype=torch.long, device=device)),
            )  # (B,)
        return result

    @abstractmethod
    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        """Implement the batch transformation in a concrete step.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
