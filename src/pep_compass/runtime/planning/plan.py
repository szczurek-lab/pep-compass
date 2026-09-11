"""Construct deterministic run plans from tasks and pipeline variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pep_compass.runtime.configuration.schema import RuntimeConfiguration
from pep_compass.runtime.planning.input import InputTask, materialize_input_tasks
from pep_compass.runtime.planning.variants import (
    PipelineVariant,
    materialize_variants,
)


@dataclass(frozen=True, slots=True)
class PlannedRun:
    """Describe one stable, independently executable plan entry."""

    index: int
    task: InputTask
    variant: PipelineVariant
    seed: int | None

    @property
    def run_id(self) -> str:
        """Return a stable identifier unique across the complete plan."""
        return f"run_{self.index:05d}"


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Contain all deterministic run entries for one runtime invocation."""

    entries: tuple[PlannedRun, ...]

    def select(self, indices: set[int] | None) -> "ExecutionPlan":
        """Return a plan restricted to requested global run indices."""
        if indices is None:
            return self
        available = {entry.index for entry in self.entries}
        missing = indices - available
        if missing:
            raise ValueError(f"Run indices are outside the plan: {sorted(missing)}")
        return ExecutionPlan(
            tuple(entry for entry in self.entries if entry.index in indices)
        )


def materialize_execution_plan(
    configuration: RuntimeConfiguration,
    *,
    working_directory: Path,
) -> ExecutionPlan:
    """Materialize tasks, variants, global indices and deterministic seeds."""
    tasks = materialize_input_tasks(
        configuration.experiment.input,
        working_directory=working_directory,
    )
    variants = materialize_variants(
        configuration.pipeline,
        configuration.experiment.grid,
    )
    entries = []
    for variant in variants:
        for task in tasks:
            index = variant.index * len(tasks) + task.index
            seed_offset = (
                task.index
                if configuration.experiment.seed_scope == "task"
                else index
            )
            base_seed = configuration.experiment.seed
            seed = base_seed + seed_offset if base_seed is not None else None
            entries.append(PlannedRun(index, task, variant, seed))
    return ExecutionPlan(tuple(entries))
