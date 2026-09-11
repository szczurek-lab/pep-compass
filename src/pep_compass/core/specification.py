"""Neutral declarations used to construct a PepCompass pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypeAlias

from pep_compass.optimization.engine.execution.state import OptimizationLimits


ComponentKind = Literal["walker", "mutation_generator", "filter", "oracle"]
ParallelExecution = Literal["sequential", "concurrent"]
TrajectoryExecution = Literal["sequential", "batched"]


@dataclass(frozen=True, slots=True)
class ComponentSpecification:
    """Declare one registered computational component."""

    kind: ComponentKind
    method: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FlowSpecification:
    """Declare steps executed sequentially in configuration order."""

    steps: tuple["StepSpecification", ...]


@dataclass(frozen=True, slots=True)
class LoopSpecification:
    """Declare repeated execution of one nested flow."""

    iterations: int
    body: FlowSpecification


@dataclass(frozen=True, slots=True)
class BranchSpecification:
    """Declare one named parallel branch."""

    name: str
    body: FlowSpecification


@dataclass(frozen=True, slots=True)
class ParallelSpecification:
    """Declare parallel branches and their deterministic merge policy."""

    branches: tuple[BranchSpecification, ...]
    execution: ParallelExecution = "sequential"
    merge: str = "concatenate"


@dataclass(frozen=True, slots=True)
class LocalEnumerationSpecification:
    """Declare SORBES trajectories with per-position mutation filtering.

    ``walker`` advances only the private trajectory stream. ``generator`` and
    ``filters`` produce candidates appended to the result pool without feeding
    them back into the next walker iteration.
    """

    trajectories: int
    iterations: int | None
    walk_time: float | None
    walker: ComponentSpecification
    generator: ComponentSpecification
    filters: FlowSpecification
    trajectory_execution: TrajectoryExecution = "batched"
    include_walk_points: bool = True


StepSpecification: TypeAlias = (
    ComponentSpecification
    | FlowSpecification
    | LoopSpecification
    | ParallelSpecification
    | LocalEnumerationSpecification
)


@dataclass(frozen=True, slots=True)
class PipelineSpecification:
    """Declare a complete pipeline independently of YAML and runtime I/O."""

    root: FlowSpecification
    limits: OptimizationLimits = field(default_factory=OptimizationLimits)
