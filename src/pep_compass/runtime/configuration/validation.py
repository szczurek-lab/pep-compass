"""Validation of runtime concerns before plan materialization."""

from __future__ import annotations

from pep_compass.runtime.configuration.schema import RuntimeConfiguration
from pep_compass.autoencoder.registry import AutoencoderRegistry
from pep_compass.utils.strategy_factory import validate_factory_parameters
from pep_compass.registry import load_builtin_registrations


def validate_runtime_configuration(configuration: RuntimeConfiguration) -> None:
    """Validate runtime values independent of computation-graph semantics.

    :param configuration: Typed runtime configuration.
    :raises ValueError: If execution, input or persistence settings are invalid.
    """
    if not configuration.experiment.name:
        raise ValueError("experiment.name cannot be empty.")
    if configuration.experiment.seed_scope not in {"run", "task"}:
        raise ValueError("experiment.seed_scope must be run or task.")
    if configuration.execution.backend not in {"local", "subprocess", "slurm"}:
        raise ValueError("execution.backend must be local, subprocess, or slurm.")
    if (
        isinstance(configuration.execution.max_workers, bool)
        or configuration.execution.max_workers < 1
    ):
        raise ValueError("execution.max_workers must be a positive integer.")
    if configuration.tracking.level not in {"short", "normal", "all"}:
        raise ValueError("tracking.level must be short, normal, or all.")
    if configuration.tracking.candidate_snapshots not in {"none", "oracle", "all"}:
        raise ValueError(
            "tracking.candidate_snapshots must be none, oracle, or all."
        )
    if not configuration.autoencoder.method:
        raise ValueError("autoencoder.method cannot be empty.")
    if not configuration.autoencoder.model:
        raise ValueError("autoencoder.model cannot be empty.")
    _validate_autoencoder(configuration)


def _validate_autoencoder(configuration: RuntimeConfiguration) -> None:
    """Validate autoencoder method, named model and factory parameters."""
    load_builtin_registrations()

    autoencoder = configuration.autoencoder
    descriptor = AutoencoderRegistry.model(autoencoder.method, autoencoder.model)
    parameters = {
        **descriptor.parameters,
        **dict(autoencoder.parameters),
        "device": autoencoder.device,
    }
    validate_factory_parameters(
        AutoencoderRegistry.method(autoencoder.method),
        parameters,
    )
