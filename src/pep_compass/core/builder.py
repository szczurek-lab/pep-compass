"""Construct executable PepCompass pipelines from neutral specifications."""

from __future__ import annotations

from pep_compass.autoencoder.base import Autoencoder
from pep_compass.core.specification import (
    ComponentSpecification,
    FlowSpecification,
    LoopSpecification,
    LocalEnumerationSpecification,
    ParallelSpecification,
    PipelineSpecification,
    StepSpecification,
)
from pep_compass.core.estimation import estimate_pipeline_stability
from pep_compass.core.validation import (
    validate_pipeline_specification,
    validate_registered_components,
)
from pep_compass.optimization.engine import (
    Flow,
    LocalEnumeration,
    Loop,
    Parallel,
    build_merger,
)
from pep_compass.optimization.pipeline import PepCompassPipeline
from pep_compass.optimization.stability_estimation.monitoring import (
    NullStabilityMonitor,
    StabilityMonitor,
)
from pep_compass.optimization.engine.execution.step import Step
from pep_compass.optimization.tracking import StepTracker
from pep_compass.utils.logger import get_custom_logger
from pep_compass.registry import component_catalog, load_builtin_registrations

logger = get_custom_logger(__name__)


class PipelineBuilder:
    """Resolve a specification into the executable ``Step`` object tree.

    Runtime parses YAML into neutral specification nodes before calling this
    builder. Composite nodes become engine operations. Component nodes select
    a family manager by ``kind`` and a registered factory by ``method``. The
    resulting root object stores the complete execution order; no scheduler
    looks up the next component while the pipeline is running.

    :param autoencoder: Initialized autoencoder injected into components that
        declare this service.
    """

    def __init__(self, autoencoder: Autoencoder) -> None:
        self.autoencoder = autoencoder
        self._load_builtin_components()

    def build(
        self,
        specification: PipelineSpecification,
        *,
        tracker: StepTracker | None = None,
        stability_monitor: StabilityMonitor | NullStabilityMonitor | None = None,
        initial_candidates: int | None = None,
    ) -> PepCompassPipeline:
        """Construct one executable pipeline from a neutral declaration.

        :param specification: Validatable computation-graph declaration.
        :param tracker: Optional runtime tracker.
        :param stability_monitor: Optional memory monitor.
        :return: Fully initialized executable pipeline.
        """
        validate_pipeline_specification(specification)
        validate_registered_components(specification)
        root = self._build_step(specification.root)
        stability_monitor = stability_monitor or NullStabilityMonitor()
        if isinstance(stability_monitor, StabilityMonitor):
            stability_monitor.configure_budgets(
                iterations=_declared_iterations(specification.root),
                oracle_calls=specification.limits.oracle_calls,
                generated_candidates=specification.limits.generated_candidates,
            )
        if initial_candidates is not None:
            estimate = estimate_pipeline_stability(
                specification,
                input_candidates=initial_candidates,
                latent_dimension=self.autoencoder.latent_dim,
            )
            logger.info(
                "Pipeline stability input_candidates=%s output_upper=%s "
                "peak_upper=%s latent_bytes_upper=%s warnings=%s.",
                estimate.input_candidates,
                estimate.output_candidates_upper,
                estimate.peak_candidates_upper,
                estimate.latent_bytes_upper,
                estimate.warnings,
            )
        logger.info("Constructed PepCompass pipeline root=%s.", root.name)
        return PepCompassPipeline(
            autoencoder=self.autoencoder,
            root=root,
            tracker=tracker,
            limits=specification.limits,
            stability_monitor=stability_monitor,
        )

    @staticmethod
    def _load_builtin_components() -> None:
        """Load built-in component registrations before resolution."""
        load_builtin_registrations()

    def _build_step(self, specification: StepSpecification) -> Step:
        """Recursively construct one declared computation node."""
        # Leaf component resolved through its family registry
        if isinstance(specification, ComponentSpecification):
            return self._build_component(specification)

        # Composite operations retaining their configured child order
        if isinstance(specification, FlowSpecification):
            return Flow([self._build_step(step) for step in specification.steps])
        if isinstance(specification, LoopSpecification):
            return Loop(
                self._build_step(specification.body),
                specification.iterations,
            )
        if isinstance(specification, ParallelSpecification):
            return Parallel(
                {
                    branch.name: self._build_step(branch.body)
                    for branch in specification.branches
                },
                execution=specification.execution,
                merger=build_merger(specification.merge),
            )
        if isinstance(specification, LocalEnumerationSpecification):
            return LocalEnumeration(
                walker=self._build_component(specification.walker),
                mutation_generator=self._build_component(specification.generator),
                filters=self._build_step(specification.filters),
                trajectories=specification.trajectories,
                trajectory_execution=specification.trajectory_execution,
                iterations=specification.iterations,
                walk_time=specification.walk_time,
                include_walk_points=specification.include_walk_points,
            )
        raise TypeError(f"Unsupported pipeline specification: {specification!r}")

    def _build_component(self, specification: ComponentSpecification) -> Step:
        """Resolve a component family, registered method, and injected services.

        ``kind`` selects the registry and ``method`` selects the factory within
        that registry. Construction occurs once while building the pipeline;
        execution later follows the already constructed ``Step`` tree.
        """
        # Component family and strategy selected through the shared catalog.
        parameters = dict(specification.parameters)
        services = {"autoencoder": self.autoencoder}
        return component_catalog.build(
            specification.kind,
            specification.method,
            parameters,
            services=services,
        )


def _declared_iterations(specification: StepSpecification) -> int | None:
    """Return a static upper bound for engine-controlled iterations."""
    if isinstance(specification, ComponentSpecification):
        return 0
    if isinstance(specification, FlowSpecification):
        values = [_declared_iterations(step) for step in specification.steps]
        return sum(values) if all(value is not None for value in values) else None
    if isinstance(specification, LoopSpecification):
        body = _declared_iterations(specification.body)
        return None if body is None else specification.iterations * (body + 1)
    if isinstance(specification, ParallelSpecification):
        values = [_declared_iterations(branch.body) for branch in specification.branches]
        return sum(values) if all(value is not None for value in values) else None
    if isinstance(specification, LocalEnumerationSpecification):
        if specification.iterations is None:
            return None
        return specification.trajectories * specification.iterations
    raise TypeError(f"Unsupported pipeline specification: {specification!r}")
