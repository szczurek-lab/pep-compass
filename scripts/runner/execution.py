"""Task execution for local processes and Slurm steps."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from builders import build_black_box, build_optimizer

logger = logging.getLogger(__name__)

OBJECTIVE_DESCRIPTIONS = {
    "apex": "Configured aggregate of log2 predicted MIC values.",
    "battleamp": "Log2 BattleAMP prediction; lower values are optimized.",
    "clasp": (
        "Log2 predicted MIC plus lambda times the MEROPS cleavage potential; "
        "lower values are optimized."
    ),
    "hydrophobicity": "Predicted peptide hydrophobicity on the configured scale.",
    "toxipep": "ToxiPep model prediction; lower values are optimized.",
}


def run_task(task: dict[str, Any]) -> None:
    """Construct and execute one fully resolved optimization task.

    :param task: Serialized task containing configuration, sequence, seed, and
        output metadata.
    """
    import numpy as np

    from pep_compass.optimization.black_box.negative_black_box import NegativeBlackBox

    config = task["config"]
    black_box = build_black_box(config)
    optimizer, observed_black_box, encoder_decoder = build_optimizer(config, black_box)
    if config["optimizer"]["name"] == "lebo":
        from pep_compass.optimization.lebo.trajectory_tracking import LeboCSVTracker

        objective_info = observed_black_box.get_black_box_info()
        tracking = config["tracking"]
        optimizer.tracker = LeboCSVTracker(
            output_directory=(
                Path(task["output_path"]) / "tracking" / task["experiment_id"]
            ),
            run_id=task["task_id"],
            level=tracking["level"],
            candidate_strategy=config["optimizer"]["lebo"]["candidate_strategy"],
            objective_name=objective_info.name,
            objective_direction=(
                "maximize" if observed_black_box.maximize else "minimize"
            ),
            objective_description=OBJECTIVE_DESCRIPTIONS[config["black_box"]["name"]],
            objective_parameters=config["black_box"],
            encoder_decoder=encoder_decoder,
            store_latents=tracking["store_latents"],
            component_provider=getattr(
                observed_black_box, "score_components_for", None
            ),
            component_fields=getattr(observed_black_box, "component_fields", ()),
        )
    else:
        from pep_compass.optimization.black_box.csv_observer import CSVObserver

        observer = CSVObserver(maximize=observed_black_box.maximize)
        observed_black_box.set_observer(observer)
        observer.initialize_observer(
            observed_black_box.get_black_box_info(),
            {
                "experiment_id": task["experiment_id"],
                "experiment_path": task["output_path"],
            },
            task["seed"],
            encoder_decoder=encoder_decoder,
        )
    logger.info("Starting %s", task["experiment_id"])
    if config["optimizer"]["name"] == "lambo2":
        target = (
            NegativeBlackBox(black_box)
            if config["optimizer"]["lambo2"]["negate_objective"]
            else black_box
        )
        solver = optimizer(
            black_box=target,
            x0=np.array([list(task["sequence"])], dtype="<U1"),
        )
        solver.solve(max_iter=config["optimizer"]["lambo2"]["iterations"])
    else:
        optimizer.optimize(
            evaluation_budget=config["evaluation_budget"],
            starting_point=task["sequence"],
            rng_seed=task["seed"],
        )


def run_task_file(path: Path) -> None:
    """Load and execute one materialized task JSON file.

    :param path: Task JSON produced by :func:`_prepare_tasks`.
    """
    with path.open(encoding="utf-8") as task_file:
        run_task(json.load(task_file))


def srun_command(task_path: Path, execution: dict[str, Any]) -> list[str]:
    """Build the isolated ``srun`` command for one task.

    :param task_path: Materialized task JSON file.
    :param execution: Execution subsection of the resolved configuration.
    :return: Argument vector suitable for :func:`subprocess.run`.
    """
    srun = execution["srun"]
    return [
        srun["command"],
        *srun["arguments"],
        sys.executable,
        str(Path(__file__).with_name("run_optimization.py").resolve()),
        "--task-file",
        str(task_path.resolve()),
    ]


def run_srun_tasks(task_paths: list[Path], execution: dict[str, Any]) -> None:
    """Submit isolated srun steps while enforcing the configured concurrency."""
    commands = [srun_command(task_path, execution) for task_path in task_paths]

    run_commands(commands, execution["max_parallel_runs"])


def run_commands(commands: list[list[str]], max_parallel_runs: int) -> None:
    """Execute isolated task commands with bounded concurrency."""

    def execute(command: list[str]) -> None:
        """Execute one child command and propagate non-zero exit status."""
        logger.info("Executing %s", " ".join(command))
        subprocess.run(command, check=True)

    with ThreadPoolExecutor(max_workers=max_parallel_runs) as executor:
        futures = [executor.submit(execute, command) for command in commands]
        for future in futures:
            future.result()


def run_local_tasks(
    tasks: list[dict[str, Any]],
    task_paths: list[Path],
    execution: dict[str, Any],
) -> None:
    """Run one task in-process or multiple tasks in isolated Python processes."""
    if execution["max_parallel_runs"] == 1:
        for task in tasks:
            run_task(task)
        return
    commands = [
        [
            sys.executable,
            str(Path(__file__).with_name("run_optimization.py").resolve()),
            "--task-file",
            str(task_path.resolve()),
        ]
        for task_path in task_paths
    ]
    run_commands(commands, execution["max_parallel_runs"])

