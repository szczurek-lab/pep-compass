"""Load JSON or YAML into the typed runtime configuration schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from pep_compass.runtime.configuration.schema import (
    AutoencoderConfiguration,
    ExecutionConfiguration,
    ExperimentConfiguration,
    RuntimeConfiguration,
    TrackingConfiguration,
)
from pep_compass.runtime.configuration.validation import validate_runtime_configuration


def load_runtime_configuration(path: str | Path) -> RuntimeConfiguration:
    """Load and validate one PepCompass runtime configuration.

    :param path: JSON, YAML or YML configuration path.
    :return: Typed runtime configuration.
    :raises ValueError: If the document or configuration is invalid.
    """
    source_path = Path(path).resolve()
    with source_path.open(encoding="utf-8") as stream:
        if source_path.suffix.lower() == ".json":
            raw = json.load(stream)
        elif source_path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(stream)
        else:
            raise ValueError("Configuration must use JSON, YAML, or YML format.")
    if not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be a mapping.")
    configuration = _parse_runtime_configuration(raw, source_path)
    validate_runtime_configuration(configuration)
    return configuration


def _parse_runtime_configuration(
    raw: Mapping[str, Any],
    source_path: Path,
) -> RuntimeConfiguration:
    """Convert a raw mapping into immutable runtime configuration objects."""
    experiment_raw = _mapping(raw.get("experiment"), "experiment")
    autoencoder_raw = _mapping(raw.get("autoencoder"), "autoencoder")
    pipeline_raw = _mapping(raw.get("pipeline"), "pipeline")
    tracking_raw = _mapping(raw.get("tracking", {}), "tracking")
    execution_raw = _mapping(raw.get("execution", {}), "execution")
    output_raw = experiment_raw.get("output_directory")
    grid_raw = _mapping(experiment_raw.get("grid", {}), "experiment.grid")
    return RuntimeConfiguration(
        experiment=ExperimentConfiguration(
            name=str(experiment_raw.get("name", "")),
            input=_mapping(experiment_raw.get("input"), "experiment.input"),
            output_directory=Path(output_raw) if output_raw is not None else None,
            seed=experiment_raw.get("seed"),
            seed_scope=experiment_raw.get("seed_scope", "run"),
            grid={key: tuple(value) for key, value in grid_raw.items()},
        ),
        autoencoder=AutoencoderConfiguration(
            method=str(autoencoder_raw.get("method", "")),
            model=str(autoencoder_raw.get("model", "")),
            device=str(autoencoder_raw.get("device", "cpu")),
            parameters=_mapping(
                autoencoder_raw.get("parameters", {}),
                "autoencoder.parameters",
            ),
        ),
        pipeline=pipeline_raw,
        tracking=TrackingConfiguration(
            level=tracking_raw.get("level", "normal"),
            max_depth=tracking_raw.get("max_depth"),
            store_latents=bool(tracking_raw.get("store_latents", False)),
            store_fields=bool(tracking_raw.get("store_fields", False)),
            field_names=(
                tuple(tracking_raw["field_names"])
                if tracking_raw.get("field_names") is not None
                else None
            ),
            candidate_snapshots=tracking_raw.get("candidate_snapshots", "none"),
            monitor_stability=bool(tracking_raw.get("monitor_stability", True)),
        ),
        execution=ExecutionConfiguration(
            backend=execution_raw.get("backend", "local"),
            max_workers=execution_raw.get("max_workers", 1),
            resume=bool(execution_raw.get("resume", False)),
            continue_on_error=bool(execution_raw.get("continue_on_error", False)),
            slurm=_mapping(execution_raw.get("slurm", {}), "execution.slurm"),
        ),
        source_path=source_path,
    )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Require and return a mapping at a configuration path."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return value
