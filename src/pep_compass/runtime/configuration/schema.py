"""Typed in-memory schema of PepCompass runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


@dataclass(frozen=True, slots=True)
class AutoencoderConfiguration:
    """Select an autoencoder implementation and named model variant."""

    method: str
    model: str
    device: str = "cpu"
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrackingConfiguration:
    """Control written step data and runtime stability monitoring."""

    level: Literal["short", "normal", "all"] = "normal"
    max_depth: int | None = None
    store_latents: bool = False
    store_fields: bool = False
    field_names: tuple[str, ...] | None = None
    candidate_snapshots: Literal["none", "oracle", "all"] = "none"
    monitor_stability: bool = True


@dataclass(frozen=True, slots=True)
class ExecutionConfiguration:
    """Control where and how independent plan entries execute."""

    backend: Literal["local", "subprocess", "slurm"] = "local"
    max_workers: int = 1
    resume: bool = False
    continue_on_error: bool = False
    slurm: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExperimentConfiguration:
    """Describe experiment identity, input, output and variant expansion."""

    name: str
    input: Mapping[str, Any]
    output_directory: Path | None = None
    seed: int | None = None
    seed_scope: Literal["run", "task"] = "run"
    grid: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    """Contain all information required to materialize and execute a plan."""

    experiment: ExperimentConfiguration
    autoencoder: AutoencoderConfiguration
    pipeline: Mapping[str, Any]
    tracking: TrackingConfiguration = field(default_factory=TrackingConfiguration)
    execution: ExecutionConfiguration = field(default_factory=ExecutionConfiguration)
    source_path: Path | None = None
