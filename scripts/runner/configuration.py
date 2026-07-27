"""Configuration loading, validation, grid expansion, and task preparation."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

SUPPORTED_BLACK_BOXES = {"apex", "battleamp", "clasp", "hydrophobicity", "toxipep"}
SUPPORTED_CANDIDATE_STRATEGIES = {
    "lebo",
    "lpbebo",
    "lams",
    "tandem",
    "move",
    "random_walker",
    "random_mutang",
}
SUPPORTED_OPTIMIZERS = {"lebo", "random_mutation", "cmaes", "saasbo", "lambo2"}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a child configuration into its parent."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    """Load JSON configuration and resolve relative inheritance and CSV paths."""
    path = path.resolve()
    visited = visited or set()
    if path in visited:
        raise ValueError(f"Circular config inheritance detected at {path}")
    visited.add(path)
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    parent = config.pop("extends", None)
    if parent is not None:
        config = deep_merge(load_config(path.parent / parent, visited), config)
    if config.get("input_csv") is not None:
        config["input_csv"] = str((path.parent / config["input_csv"]).resolve())
    return config


def load_sequences(path: Path) -> list[dict[str, Any]]:
    """Read named starting peptides and optional per-sequence repetitions."""
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"name", "sequence"}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError("Input CSV must contain name and sequence columns")
        sequences = []
        names = set()
        for row_number, row in enumerate(reader, start=2):
            name = row["name"].strip()
            sequence = row["sequence"].strip().upper()
            repetitions = int((row.get("repetitions") or "1").strip())
            if not name or not sequence:
                raise ValueError(f"Empty name or sequence in CSV row {row_number}")
            if name in names:
                raise ValueError(f"Duplicate sequence name in input CSV: {name}")
            if repetitions < 1:
                raise ValueError(
                    f"Repetitions must be positive in CSV row {row_number}"
                )
            names.add(name)
            sequences.append(
                {"name": name, "sequence": sequence, "repetitions": repetitions}
            )
    if not sequences:
        raise ValueError("Input CSV must contain at least one sequence")
    return sequences


def set_nested_value(config: dict[str, Any], path: str, value: Any) -> None:
    """Assign a value through a dotted configuration path."""
    keys = path.split(".")
    target = config
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"Grid path does not reference a config object: {path}")
        target = child
    target[keys[-1]] = value


def expand_grid(config: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Expand explicit dotted-path value lists into a Cartesian parameter grid."""
    base = deepcopy(config)
    grid = base.pop("grid", {})
    if not isinstance(grid, dict):
        raise ValueError("grid must map dotted config paths to value lists")
    for path, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid values for {path} must be a non-empty list")
    if not grid:
        return [({}, base)]
    paths = list(grid)
    expanded = []
    for values in product(*(grid[path] for path in paths)):
        parameters = dict(zip(paths, values))
        variant = deepcopy(base)
        for path, value in parameters.items():
            set_nested_value(variant, path, value)
        expanded.append((parameters, variant))
    return expanded


def validate_config(config: dict[str, Any]) -> None:
    """Validate component registries, tracking, and execution settings."""
    optimizer_name = config["optimizer"]["name"]
    black_box_name = config["black_box"]["name"]
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")
    if black_box_name not in SUPPORTED_BLACK_BOXES:
        raise ValueError(f"Unsupported black box: {black_box_name!r}")
    if optimizer_name == "lebo":
        candidate_strategy = config["optimizer"]["lebo"]["candidate_strategy"]
        if candidate_strategy not in SUPPORTED_CANDIDATE_STRATEGIES:
            raise ValueError(
                f"Unsupported LE-BO candidate strategy: {candidate_strategy!r}"
            )
    tracking = config["tracking"]
    if tracking["level"] not in {"short", "normal", "full"}:
        raise ValueError("tracking.level must be short, normal, or full")
    if not isinstance(tracking["store_latents"], bool):
        raise ValueError("tracking.store_latents must be a boolean")
    if config["execution"]["backend"] not in {"local", "srun"}:
        raise ValueError("execution.backend must be 'local' or 'srun'")
    if config["execution"]["max_parallel_runs"] < 1:
        raise ValueError("execution.max_parallel_runs must be positive")
    devices = config["execution"].get("devices", [])
    if not isinstance(devices, list) or not all(
        isinstance(device, str) and device for device in devices
    ):
        raise ValueError("execution.devices must be a list of non-empty device names")


def candidate_strategy_label(config: dict[str, Any]) -> str:
    """Return the candidate-strategy label only for LE-BO tasks."""
    if config["optimizer"]["name"] != "lebo":
        return ""
    return config["optimizer"]["lebo"]["candidate_strategy"]


def prepare_tasks(
    config: dict[str, Any],
    sequences: list[dict[str, Any]],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Materialize grid variants, isolated task files, and manifests."""
    variants = expand_grid(config)
    task_directory = output_root / "tasks"
    task_directory.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    task_paths: list[Path] = []
    base_seed = config["seed"]
    with (output_root / "grid_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as grid_file, (output_root / "run_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as run_file:
        grid_writer = csv.DictWriter(
            grid_file, fieldnames=["grid_id", "output_path", "parameters"]
        )
        run_writer = csv.DictWriter(
            run_file,
            fieldnames=[
                "task_id", "grid_id", "name", "sequence", "repetition",
                "seed", "device", "output_path", "task_file",
            ],
        )
        grid_writer.writeheader()
        run_writer.writeheader()
        task_index = 0
        for grid_index, (parameters, variant) in enumerate(variants):
            validate_config(variant)
            grid_id = f"grid_{grid_index:04d}"
            variant_path = output_root / grid_id
            variant_path.mkdir(parents=True, exist_ok=True)
            variant["output_path"] = str(variant_path)
            with (variant_path / "resolved_config.json").open(
                "w", encoding="utf-8"
            ) as resolved_file:
                json.dump(variant, resolved_file, indent=2)
            grid_writer.writerow(
                {
                    "grid_id": grid_id,
                    "output_path": variant_path,
                    "parameters": json.dumps(parameters, sort_keys=True),
                }
            )
            for peptide in sequences:
                for repetition in range(peptide["repetitions"]):
                    task_id = f"task_{task_index:06d}"
                    seed = base_seed + task_index
                    experiment_id = (
                        f"{variant['optimizer']['name']}_"
                        f"{candidate_strategy_label(variant) or 'na'}_"
                        f"{peptide['name']}_r{repetition}_{seed}"
                    )
                    task_config = deepcopy(variant)
                    devices = variant["execution"].get("devices", [])
                    if devices:
                        task_config["device"] = devices[task_index % len(devices)]
                    task = {
                        "task_id": task_id,
                        "grid_id": grid_id,
                        "name": peptide["name"],
                        "sequence": peptide["sequence"],
                        "repetition": repetition,
                        "seed": seed,
                        "experiment_id": experiment_id,
                        "output_path": str(variant_path),
                        "config": task_config,
                    }
                    task_path = task_directory / f"{task_id}.json"
                    with task_path.open("w", encoding="utf-8") as task_file:
                        json.dump(task, task_file, indent=2)
                    tasks.append(task)
                    task_paths.append(task_path)
                    run_writer.writerow(
                        {
                            "task_id": task_id,
                            "grid_id": grid_id,
                            "name": peptide["name"],
                            "sequence": peptide["sequence"],
                            "repetition": repetition,
                            "seed": seed,
                            "device": task_config["device"],
                            "output_path": variant_path,
                            "task_file": task_path,
                        }
                    )
                    task_index += 1
    return tasks, task_paths
