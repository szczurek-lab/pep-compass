"""Translate pipeline configuration mappings into neutral core declarations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pep_compass.core.specification import (
    BranchSpecification,
    ComponentSpecification,
    FlowSpecification,
    LoopSpecification,
    LocalEnumerationSpecification,
    ParallelSpecification,
    PipelineSpecification,
    StepSpecification,
)
from pep_compass.optimization.engine.execution.state import OptimizationLimits
from pep_compass.registry import component_catalog, load_builtin_registrations


def parse_pipeline_specification(
    configuration: Mapping[str, Any],
) -> PipelineSpecification:
    """Parse a runtime pipeline mapping into a neutral core specification."""
    load_builtin_registrations()
    limits = _mapping(configuration.get("limits", {}), "pipeline.limits")
    steps = _sequence(configuration.get("steps"), "pipeline.steps")
    return PipelineSpecification(
        root=_parse_flow(steps, "pipeline.steps"),
        limits=OptimizationLimits(
            oracle_calls=limits.get("oracle_calls"),
            generated_candidates=limits.get("generated_candidates"),
        ),
    )


def _parse_flow(
    configurations: Sequence[Any],
    path: str,
) -> FlowSpecification:
    """Parse a sequence of single-operation mappings."""
    return FlowSpecification(
        tuple(
            _parse_step(configuration, f"{path}.{index}")
            for index, configuration in enumerate(configurations)
        )
    )


def _parse_step(configuration: Any, path: str) -> StepSpecification:
    """Parse one configured operation recursively."""
    mapping = _mapping(configuration, path)
    if len(mapping) != 1:
        raise ValueError(f"{path} must contain exactly one operation key.")
    operation, raw_settings = next(iter(mapping.items()))
    settings = _mapping(raw_settings, f"{path}.{operation}")
    if operation in component_catalog.names():
        method = settings.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError(f"{path}.{operation}.method cannot be empty.")
        return ComponentSpecification(
            kind=operation,
            method=method,
            parameters=_mapping(
                settings.get("parameters", {}),
                f"{path}.{operation}.parameters",
            ),
        )
    if operation == "loop":
        iterations = settings.get("iterations")
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise ValueError(f"{path}.loop.iterations must be an integer.")
        return LoopSpecification(
            iterations,
            _parse_flow(
                _sequence(settings.get("steps"), f"{path}.loop.steps"),
                f"{path}.loop.steps",
            ),
        )
    if operation == "parallel":
        return _parse_parallel(settings, f"{path}.parallel")
    if operation == "local_enumeration":
        return _parse_local_enumeration(settings, f"{path}.local_enumeration")
    raise ValueError(f"Unknown pipeline operation at {path}: {operation}")


def _parse_local_enumeration(
    settings: Mapping[str, Any],
    path: str,
) -> LocalEnumerationSpecification:
    """Parse trajectory-local SORBES and mutation-processing declarations."""
    trajectories = settings.get("trajectories", 1)
    iterations = settings.get("iterations")
    walk_time = settings.get("walk_time")
    walker = _parse_step({"walker": settings.get("walker")}, f"{path}.walker")
    generator = _parse_step(
        {"mutation_generator": settings.get("mutation_generator")},
        f"{path}.mutation_generator",
    )
    filters = _parse_flow(
        [
            {"filter": filter_configuration}
            for filter_configuration in _sequence(
                settings.get("filters", []),
                f"{path}.filters",
            )
        ],
        f"{path}.filters",
    )
    assert isinstance(walker, ComponentSpecification)
    assert isinstance(generator, ComponentSpecification)
    return LocalEnumerationSpecification(
        trajectories=trajectories,
        iterations=iterations,
        walk_time=walk_time,
        walker=walker,
        generator=generator,
        filters=filters,
        trajectory_execution=settings.get("trajectory_execution", "batched"),
        include_walk_points=settings.get("include_walk_points", True),
    )


def _parse_parallel(
    settings: Mapping[str, Any],
    path: str,
) -> ParallelSpecification:
    """Parse named branches or indexed replicas into explicit branches."""
    replicas = settings.get("replicas")
    raw_branches = settings.get("branches")
    if replicas is not None and raw_branches is not None:
        raise ValueError(f"{path} cannot combine replicas and branches.")
    branches: list[BranchSpecification] = []
    if replicas is not None:
        if not isinstance(replicas, int) or isinstance(replicas, bool) or replicas < 1:
            raise ValueError(f"{path}.replicas must be a positive integer.")
        steps = _sequence(settings.get("steps"), f"{path}.steps")
        branches = [
            BranchSpecification(
                f"replica_{index:03d}",
                _parse_flow(steps, f"{path}.steps"),
            )
            for index in range(replicas)
        ]
    else:
        for index, raw_branch in enumerate(
            _sequence(raw_branches, f"{path}.branches")
        ):
            branch = _mapping(raw_branch, f"{path}.branches.{index}")
            name = str(branch.get("name", f"branch_{index}"))
            branches.append(
                BranchSpecification(
                    name,
                    _parse_flow(
                        _sequence(
                            branch.get("steps"),
                            f"{path}.branches.{index}.steps",
                        ),
                        f"{path}.branches.{index}.steps",
                    ),
                )
            )
    return ParallelSpecification(
        tuple(branches),
        execution=settings.get("execution", "sequential"),
        merge=str(settings.get("merge", "concatenate")),
    )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Require and return a mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    """Require and return a non-string sequence."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be a sequence.")
    return value
