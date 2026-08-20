"""Semantic validation of neutral PepCompass pipeline specifications."""

from __future__ import annotations

from dataclasses import dataclass

from pep_compass.core.specification import (
    ComponentSpecification,
    FlowSpecification,
    LoopSpecification,
    LocalEnumerationSpecification,
    ParallelSpecification,
    PipelineSpecification,
    StepSpecification,
)


@dataclass(frozen=True, slots=True)
class PipelineDiagnostic:
    """Describe a statically detectable pipeline risk or semantic collision."""

    severity: str
    code: str
    path: str
    message: str


def validate_pipeline_specification(specification: PipelineSpecification) -> None:
    """Validate a complete pipeline declaration before object construction.

    :param specification: Neutral pipeline declaration.
    :raises ValueError: If limits, component declarations or graph nodes are invalid.
    """
    limits = specification.limits
    for name, value in (
        ("oracle_calls", limits.oracle_calls),
        ("generated_candidates", limits.generated_candidates),
    ):
        if value is not None and (isinstance(value, bool) or value < 0):
            raise ValueError(f"Pipeline limit {name} must be null or non-negative.")
    _validate_step(specification.root, "pipeline.root")


def validate_registered_components(specification: PipelineSpecification) -> None:
    """Validate registered methods and parameters without constructing models."""
    from pep_compass.registry import load_builtin_registrations

    load_builtin_registrations()

    _validate_registered_step(specification.root)


def diagnose_pipeline_specification(
    specification: PipelineSpecification,
) -> tuple[PipelineDiagnostic, ...]:
    """Return non-fatal data-flow and resource diagnostics for a valid graph."""
    diagnostics: list[PipelineDiagnostic] = []
    _diagnose_step(specification.root, "pipeline", diagnostics)
    if specification.limits.generated_candidates is not None:
        diagnostics.append(
            PipelineDiagnostic(
                severity="INFO",
                code="POST_GENERATION_LIMIT",
                path="pipeline.limits.generated_candidates",
                message=(
                    "The limit stops subsequent iterations after generation; "
                    "it does not bound one mutation product before allocation."
                ),
            )
        )
    return tuple(diagnostics)


def _diagnose_step(
    specification: StepSpecification,
    path: str,
    diagnostics: list[PipelineDiagnostic],
) -> None:
    """Inspect one graph node for known composition hazards."""
    if isinstance(specification, ComponentSpecification):
        if (
            specification.kind == "mutation_generator"
            and specification.parameters.get("maximum_candidates") is None
        ):
            diagnostics.append(
                PipelineDiagnostic(
                    severity="ERROR",
                    code="UNBOUNDED_EXPANSION",
                    path=path,
                    message=(
                        f"Mutation generator '{specification.method}' has no "
                        "maximum_candidates; its output and peak memory cannot "
                        "be bounded statically."
                    ),
                )
            )
        return
    if isinstance(specification, FlowSpecification):
        for index, step in enumerate(specification.steps):
            _diagnose_step(step, f"{path}.steps[{index}]", diagnostics)
        return
    if isinstance(specification, LoopSpecification):
        kinds = _component_kinds(specification.body)
        if "mutation_generator" in kinds and "walker" in kinds:
            diagnostics.append(
                PipelineDiagnostic(
                    severity="ERROR",
                    code="TRAJECTORY_FEEDBACK",
                    path=path,
                    message=(
                        "This generic loop contains both a walker and a mutation "
                        "generator. The complete mutation batch becomes the next "
                        "iteration's walker input. Use local_enumeration to keep "
                        "the trajectory and candidate pool separate."
                    ),
                )
            )
        _diagnose_step(specification.body, f"{path}.body", diagnostics)
        return
    if isinstance(specification, ParallelSpecification):
        if specification.execution == "concurrent":
            diagnostics.append(
                PipelineDiagnostic(
                    severity="WARNING",
                    code="SHARED_MUTABLE_STATE",
                    path=path,
                    message=(
                        "Concurrent branches share OptimizationState; counter and "
                        "trust-region updates are not transactionally isolated."
                    ),
                )
            )
        for branch in specification.branches:
            _diagnose_step(
                branch.body,
                f"{path}.branch[{branch.name}]",
                diagnostics,
            )
        return
    if isinstance(specification, LocalEnumerationSpecification):
        _diagnose_step(
            specification.generator,
            f"{path}.mutation_generator",
            diagnostics,
        )
        _diagnose_step(specification.filters, f"{path}.filters", diagnostics)
        if not specification.filters.steps:
            diagnostics.append(
                PipelineDiagnostic(
                    severity="WARNING",
                    code="NO_LOCAL_FILTER",
                    path=f"{path}.filters",
                    message=(
                        "No filter reduces each local mutation batch before it is "
                        "retained in the complete candidate pool."
                    ),
                )
            )
        return
    raise TypeError(f"Unsupported pipeline specification: {specification!r}")


def _component_kinds(specification: StepSpecification) -> set[str]:
    """Return component families nested below one operation."""
    if isinstance(specification, ComponentSpecification):
        return {specification.kind}
    if isinstance(specification, FlowSpecification):
        return set().union(*(_component_kinds(step) for step in specification.steps))
    if isinstance(specification, LoopSpecification):
        return _component_kinds(specification.body)
    if isinstance(specification, ParallelSpecification):
        return set().union(
            *(_component_kinds(branch.body) for branch in specification.branches)
        )
    if isinstance(specification, LocalEnumerationSpecification):
        # LocalEnumeration owns its trajectory feedback boundary. Its walker
        # and generator must not be interpreted as siblings of an outer loop.
        return {"local_enumeration", *(_component_kinds(specification.filters))}
    raise TypeError(f"Unsupported pipeline specification: {specification!r}")


def _validate_registered_step(specification: StepSpecification) -> None:
    """Recursively validate component registry contracts."""
    if isinstance(specification, ComponentSpecification):
        from pep_compass.registry import component_catalog

        component_catalog.validate(
            specification.kind,
            specification.method,
            specification.parameters,
        )
        return
    if isinstance(specification, FlowSpecification):
        for step in specification.steps:
            _validate_registered_step(step)
        return
    if isinstance(specification, LoopSpecification):
        _validate_registered_step(specification.body)
        return
    if isinstance(specification, ParallelSpecification):
        for branch in specification.branches:
            _validate_registered_step(branch.body)
        return
    if isinstance(specification, LocalEnumerationSpecification):
        _validate_registered_step(specification.walker)
        _validate_registered_step(specification.generator)
        _validate_registered_step(specification.filters)
        return
    raise TypeError(f"Unsupported pipeline specification: {specification!r}")


def _validate_step(specification: StepSpecification, path: str) -> None:
    """Validate one node and recursively validate its children."""
    if isinstance(specification, ComponentSpecification):
        if not specification.method:
            raise ValueError(f"{path}.method cannot be empty.")
        return
    if isinstance(specification, FlowSpecification):
        for index, step in enumerate(specification.steps):
            _validate_step(step, f"{path}.steps.{index}")
        return
    if isinstance(specification, LoopSpecification):
        if specification.iterations < 0:
            raise ValueError(f"{path}.iterations must be non-negative.")
        _validate_step(specification.body, f"{path}.body")
        return
    if isinstance(specification, ParallelSpecification):
        if specification.execution not in {"sequential", "concurrent"}:
            raise ValueError(f"{path}.execution is invalid.")
        if specification.merge != "concatenate":
            raise ValueError(f"{path}.merge is not implemented.")
        if not specification.branches:
            raise ValueError(f"{path}.branches cannot be empty.")
        names = [branch.name for branch in specification.branches]
        if len(names) != len(set(names)):
            raise ValueError(f"{path}.branches contain duplicate names.")
        for index, branch in enumerate(specification.branches):
            if not branch.name:
                raise ValueError(f"{path}.branches.{index}.name cannot be empty.")
            _validate_step(branch.body, f"{path}.branches.{index}.body")
        return
    if isinstance(specification, LocalEnumerationSpecification):
        if specification.trajectories < 1:
            raise ValueError(f"{path}.trajectories must be positive.")
        if specification.trajectory_execution not in {"sequential", "batched"}:
            raise ValueError(f"{path}.trajectory_execution is invalid.")
        if (specification.iterations is None) == (specification.walk_time is None):
            raise ValueError(
                f"{path} requires exactly one of iterations or walk_time."
            )
        if specification.iterations is not None and specification.iterations < 1:
            raise ValueError(f"{path}.iterations must be positive.")
        if specification.walk_time is not None and specification.walk_time <= 0:
            raise ValueError(f"{path}.walk_time must be positive.")
        if specification.walker.kind != "walker":
            raise ValueError(f"{path}.walker must declare a walker.")
        if specification.generator.kind != "mutation_generator":
            raise ValueError(
                f"{path}.generator must declare a mutation generator."
            )
        _validate_step(specification.walker, f"{path}.walker")
        _validate_step(specification.generator, f"{path}.generator")
        _validate_step(specification.filters, f"{path}.filters")
        return
    raise TypeError(f"Unsupported pipeline specification at {path}: {specification!r}")
