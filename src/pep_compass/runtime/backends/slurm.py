"""Generate Slurm array scripts for a materialized execution plan."""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pep_compass.runtime.backends.subprocess import worker_command
from pep_compass.runtime.planning.plan import ExecutionPlan


def write_slurm_array_script(
    plan: ExecutionPlan,
    configuration_path: Path,
    output_path: Path,
    *,
    working_directory: Path,
    settings: Mapping[str, Any],
    autoencoder_device: str | None = None,
) -> Path:
    """Write, but do not submit, a Slurm array script."""
    if not plan.entries:
        raise ValueError("Cannot create a Slurm script for an empty plan.")
    indices = ",".join(str(entry.index) for entry in plan.entries)
    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --array={indices}",
        f"#SBATCH --chdir={working_directory.resolve()}",
    ]
    for key in ("job_name", "partition", "time", "gres", "cpus_per_task", "mem"):
        value = settings.get(key)
        if value is None:
            continue
        text = str(value)
        if "\n" in text or "\r" in text:
            raise ValueError(f"Invalid newline in Slurm setting: {key}")
        directives.append(f"#SBATCH --{key.replace('_', '-')}={text}")
    command = worker_command(
        configuration_path,
        "$SLURM_ARRAY_TASK_ID",
        working_directory=working_directory,
        autoencoder_device=autoencoder_device,
    )
    rendered = [shlex.quote(part) for part in command]
    token = rendered.index("'$SLURM_ARRAY_TASK_ID'")
    rendered[token] = '"$SLURM_ARRAY_TASK_ID"'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join([*directives, "", "set -euo pipefail", "", " ".join(rendered), ""]),
        encoding="utf-8",
    )
    output_path.chmod(0o755)
    return output_path
