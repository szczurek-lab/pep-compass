"""Public executable PepCompass pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.autoencoder.base import Autoencoder
from pep_compass.data_schemas.candidates import (
    CandidateBatch,
    TensorField,
)
from pep_compass.data_schemas.execution import OptimizationLimits
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.state import OptimizationState
from pep_compass.optimization.stability_estimation.monitoring import (
    NullStabilityMonitor,
    StabilityMonitor,
)
from pep_compass.optimization.tracking import NullStepTracker, StepTracker
from pep_compass.steps.base import Step
from pep_compass.utils.logger import get_custom_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal, Self

    from pep_compass.core.wrapper.visualization import GraphView
    from pep_compass.data_schemas.execution import RunnerSettings
    from pep_compass.data_schemas.graph import PipelineSchema
    from pep_compass.data_schemas.results import RunResult


logger = get_custom_logger(__name__)


class ExecutablePipeline:
    """Execute a fully constructed PepCompass computation.

    The pipeline is independent of YAML, experiment planning, persistence and
    CLI concerns. Callers may construct it manually or obtain it from
    :class:`pep_compass.core.builder.PipelineBuilder`.

    :param autoencoder: Autoencoder shared by computational components.
    :param root: Root operation of the executable computation graph.
    :param tracker: Optional detailed step tracker.
    :param limits: Optional run-level computation limits.
    :param stability_monitor: Optional process and accelerator memory monitor.
    """

    def __init__(
        self,
        *,
        autoencoder: Autoencoder,
        schema: PipelineSchema | None,
        model_manifest: Sequence[Mapping[str, object]] = (),
        settings: RunnerSettings | None = None,
        root: Step,
        tracker: StepTracker | None = None,
        limits: OptimizationLimits | None = None,
        stability_monitor: StabilityMonitor | NullStabilityMonitor | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder
        :param schema: Immutable declaration or its explicitly supported serialized representation.
        :type schema: PipelineSchema | None
        :param model_manifest: Model manifest.
        :type model_manifest: Sequence[Mapping[str, object]]
        :param settings: Validated settings or a resolved settings manifest.
        :type settings: RunnerSettings | None
        :param root: Root resource directory or root graph node, as declared by its type.
        :type root: Step
        :param tracker: Tracker.
        :type tracker: StepTracker | None
        :param limits: Limits.
        :type limits: OptimizationLimits | None
        :param stability_monitor: Stability monitor.
        :type stability_monitor: StabilityMonitor | NullStabilityMonitor | None
        :rtype: None
        """
        from pep_compass.data_schemas.execution import RunnerSettings

        self.runner_settings = settings or RunnerSettings()
        self.schema = schema
        self.model_manifest = tuple(model_manifest)
        self.autoencoder = autoencoder
        self.root = root
        self.tracker = tracker or NullStepTracker()
        self.limits = limits or OptimizationLimits()
        self.stability_monitor = stability_monitor or NullStabilityMonitor()

    def run(
        self,
        sequences: Sequence[str],
        *,
        settings: RunnerSettings | None = None,
        seed: int | None = None,
    ) -> RunResult:
        """Execute with fresh run state and the common saver contract.

        :param sequences: Peptide sequences in input order.
        :type sequences: Sequence[str]
        :param settings: Validated settings or a resolved settings manifest.
        :type settings: RunnerSettings | None
        :param seed: Random seed; None leaves the caller-selected unseeded behavior.
        :type seed: int | None
        :return: Result of this operation.
        :rtype: RunResult
        """
        from pep_compass.optimization.engine.runner import Runner

        settings = settings or self.runner_settings
        if seed is not None:
            settings = settings.update(execution={"seed": seed})
        return Runner(self, self.schema, settings, self.model_manifest).run(sequences)

    def set_runner_settings(
        self,
        *,
        execution: Mapping[str, object] | None = None,
        storage: Mapping[str, object] | None = None,
        results: Mapping[str, object] | None = None,
    ) -> Self:
        """Update defaults for runs performed directly on this compiled graph.

        :param execution: Execution settings or a mapping overriding their defaults.
        :type execution: Mapping[str, object] | None
        :param storage: Storage settings or a mapping overriding their defaults.
        :type storage: Mapping[str, object] | None
        :param results: Result-view settings or a mapping overriding their defaults.
        :type results: Mapping[str, object] | None
        :return: Result of this operation.
        :rtype: Self
        """
        self.runner_settings = self.runner_settings.update(
            execution=execution, storage=storage, results=results
        )
        return self

    def to_schema(
        self, format: Literal["object", "json", "text", "mermaid", "svg"] = "object"
    ) -> PipelineSchema | str:
        """Return the exact declaration snapshot compiled into this graph.

        :param format: Format.
        :type format: Literal['object', 'json', 'text', 'mermaid', 'svg']
        :return: Result of this operation.
        :rtype: PipelineSchema | str
        """
        from pep_compass.core.wrapper.visualization import render

        return render(self.schema, format=format)

    def show(self, format: Literal["text", "mermaid", "svg"] = "svg") -> GraphView:
        """Render the compiled snapshot through the common notebook graph view.

        :param format: Format.
        :type format: Literal['text', 'mermaid', 'svg']
        :return: Result of this operation.
        :rtype: GraphView
        """
        from pep_compass.core.wrapper.visualization import graph_view

        return graph_view(self.schema, format)

    def _run(
        self,
        sequences: list[str],
        *,
        seed: int | None,
        context: OptimizationContext | None = None,
        close_tracker: bool = True,
    ) -> CandidateBatch:
        """Execute one run inside the deterministic random-state boundary.

        :param sequences: Peptide sequences in input order.
        :type sequences: list[str]
        :param seed: Random seed; None leaves the caller-selected unseeded behavior.
        :type seed: int | None
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext | None
        :param close_tracker: Close tracker.
        :type close_tracker: bool
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        # Sanity check: Empty sequences
        if not sequences:
            raise ValueError("At least one starting sequence is required.")

        # Initial candidate representation
        with torch.no_grad():
            latent_origins = self.autoencoder.encode_peptides(sequences)  # (B, D)
        context = context or OptimizationContext(
            autoencoder=self.autoencoder,
            tracker=self.tracker,
            seed=seed,
            state=OptimizationState(limits=self.limits),
            stability_monitor=self.stability_monitor,
        )
        candidate_ids = torch.as_tensor(
            context.state.next_candidate_ids(len(sequences)),
            dtype=torch.long,
            device=latent_origins.device,
        )  # (B,)
        batch = CandidateBatch(sequences, latent_origins)
        batch = batch.with_field("lineage.candidate_id", TensorField(candidate_ids))
        batch = batch.with_field(
            "lineage.parent_candidate_id",
            TensorField(torch.full_like(candidate_ids, -1)),
        )

        # Computation lifecycle
        try:
            self.stability_monitor.sample("pipeline.input", batch, context.state)
            logger.info("Precomputing PepCompass pipeline steps.")
            self.root.precompute(context)
            logger.info(
                "Executing PepCompass pipeline for %s starting sequences.",
                len(sequences),
            )
            result = self.root(batch, context)
            self.stability_monitor.sample("pipeline.output", result, context.state)
            return result
        finally:
            if close_tracker:
                context.tracker.close()
