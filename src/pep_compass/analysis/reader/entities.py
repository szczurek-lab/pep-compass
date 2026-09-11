"""Domain entities describing discovered experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Describe one materialized optimization run without loading its tables.

    :param run_id: Task identifier written to tracking tables.
    :param experiment: Experiment directory name.
    :param grid_id: Grid variant identifier.
    :param tracking_path: Directory containing tracker outputs.
    :param config: Resolved run configuration.
    :param metadata: Manifest and grid values used for selection and grouping.
    """

    run_id: str
    experiment: str
    grid_id: str
    tracking_path: Path
    config: dict[str, Any] = field(compare=False, hash=False, repr=False)
    metadata: dict[str, Any] = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True, slots=True)
class Experiment:
    """Group run entities belonging to one experiment directory."""

    name: str
    root: Path
    runs: tuple[ExperimentRun, ...]


@dataclass(frozen=True, slots=True)
class ExperimentCollection:
    """Represent the path opened by the reader and all discovered experiments."""

    root: Path
    experiments: tuple[Experiment, ...]

    @property
    def runs(self) -> tuple[ExperimentRun, ...]:
        """Return all discovered runs in stable experiment order."""
        return tuple(run for experiment in self.experiments for run in experiment.runs)
