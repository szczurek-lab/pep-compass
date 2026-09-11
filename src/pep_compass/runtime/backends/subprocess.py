"""Execute plan entries in isolated local Python interpreters."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pep_compass.runtime.planning.plan import ExecutionPlan, PlannedRun


def worker_command(
    configuration_path: Path,
    run_index: int | str,
    *,
    working_directory: Path,
    autoencoder_device: str | None = None,
) -> list[str]:
    """Return the CLI command for one isolated plan entry."""
    command = [
        sys.executable,
        "-m",
        "pep_compass.runtime.cli",
        "run",
        str(configuration_path),
        "--run-index",
        str(run_index),
        "--backend",
        "local",
        "--working-directory",
        str(working_directory),
        "--worker",
    ]
    if autoencoder_device is not None:
        command.extend(("--device", autoencoder_device))
    return command


def execute_subprocess_plan(
    plan: ExecutionPlan,
    configuration_path: Path,
    *,
    working_directory: Path,
    max_workers: int,
    autoencoder_device: str | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    """Execute every selected plan entry in an isolated process."""
    if max_workers < 1:
        raise ValueError("Subprocess max_workers must be positive.")

    def execute(entry: PlannedRun) -> None:
        """Execute one isolated plan entry and propagate failure."""
        command_runner(
            worker_command(
                configuration_path,
                entry.index,
                working_directory=working_directory,
                autoencoder_device=autoencoder_device,
            ),
            check=True,
            cwd=working_directory,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(execute, plan.entries))
