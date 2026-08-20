"""Named model artifacts shared by model-backed oracle strategies."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from pep_compass.registry import (
    DirectoryArtifact,
    FileArtifact,
    GlobArtifact,
    ModelDescriptor,
    model_catalog,
)


_STRATEGIES_ROOT = Path(__file__).resolve().parent / "strategies"

_DEFAULT_MODEL_ROOTS = {
    "apex": _STRATEGIES_ROOT / "apex" / "models",
    "battleamp": _STRATEGIES_ROOT / "battleamp",
    "eipred": _STRATEGIES_ROOT / "eipred",
    "mbc_attention": _STRATEGIES_ROOT / "mbc_attention",
    "toxipep": _STRATEGIES_ROOT / "toxipep",
}

_MODEL_PROVIDER_ALIASES = {
    "apex_original": "apex",
}

_registration_lock = Lock()
_registered = False

_MODEL_DESCRIPTORS = (
    ModelDescriptor(
        provider="oracle.apex",
        name="default",
        directory="default",
        artifacts=(GlobArtifact("APEX_*", expected_count=8),),
        parameters={},
        download_script=(
            "assets/scripts/downloads/apex/download_apex_models_default.sh"
        ),
    ),
    ModelDescriptor(
        provider="oracle.apex",
        name="full",
        directory="full",
        artifacts=(GlobArtifact("trained_*", expected_count=40),),
        parameters={},
        download_script="assets/scripts/downloads/apex/download_apex_models_full.sh",
    ),
    ModelDescriptor(
        provider="oracle.battleamp",
        name="default",
        directory=".",
        artifacts=(FileArtifact("model.h5"),),
        parameters={},
    ),
    ModelDescriptor(
        provider="oracle.mbc_attention",
        name="default",
        directory=".",
        artifacts=(
            DirectoryArtifact(
                "model",
                required_files=(
                    "saved_model.pb",
                    "variables/variables.index",
                    "variables/variables.data-00000-of-00001",
                ),
            ),
        ),
        parameters={},
    ),
    ModelDescriptor(
        provider="oracle.eipred",
        name="default",
        directory="DATA",
        artifacts=(
            FileArtifact("model2.pkl/model2.pkl"),
            FileArtifact("selected_features_mrmr1000_new.csv"),
        ),
        parameters={},
    ),
    ModelDescriptor(
        provider="oracle.toxipep",
        name="default",
        directory="models",
        artifacts=(FileArtifact("best_model_0.9.pth"),),
        parameters={},
    ),
)


def register_oracle_models() -> None:
    """Register bundled oracle model descriptors without importing model stacks."""
    global _registered
    if _registered:
        return
    with _registration_lock:
        if _registered:
            return
        for descriptor in _MODEL_DESCRIPTORS:
            model_catalog.register(descriptor)
        _registered = True


def has_oracle_model_provider(strategy: str) -> bool:
    """Return whether an oracle strategy selects a named model."""
    provider_strategy = _MODEL_PROVIDER_ALIASES.get(strategy, strategy)
    return provider_strategy in _DEFAULT_MODEL_ROOTS


def resolve_oracle_model(
    strategy: str,
    model: str = "default",
    *,
    models_directory: str | Path | None = None,
) -> tuple[ModelDescriptor, tuple[Path, ...]]:
    """Resolve one named model for a model-backed oracle.

    :param strategy: Oracle strategy name and model-provider suffix.
    :param model: Registered model variant.
    :param models_directory: Optional replacement for the strategy package root.
        The replacement must preserve the descriptor-relative directory layout.
    :return: Model descriptor and validated artifact paths.
    :raises ValueError: If the strategy/model pair is unknown.
    :raises FileNotFoundError: If a required artifact is absent or incomplete.
    """
    register_oracle_models()
    provider_strategy = _MODEL_PROVIDER_ALIASES.get(strategy, strategy)
    if provider_strategy not in _DEFAULT_MODEL_ROOTS:
        raise ValueError(f"Oracle strategy has no registered models: {strategy!r}.")
    root = (
        Path(models_directory)
        if models_directory is not None
        else _DEFAULT_MODEL_ROOTS[provider_strategy]
    )
    return model_catalog.resolve(f"oracle.{provider_strategy}", model, root=root)


def validate_oracle_model(strategy: str, model: str = "default") -> None:
    """Validate a configured oracle model name without touching its artifacts.

    :param strategy: Registered oracle strategy.
    :param model: Configured named model variant.
    :raises ValueError: If the strategy has no model provider or name is unknown.
    """
    register_oracle_models()
    provider_strategy = _MODEL_PROVIDER_ALIASES.get(strategy, strategy)
    if provider_strategy not in _DEFAULT_MODEL_ROOTS:
        raise ValueError(f"Oracle strategy has no registered models: {strategy!r}.")
    model_catalog.descriptor(f"oracle.{provider_strategy}", model)


register_oracle_models()


__all__ = [
    "register_oracle_models",
    "has_oracle_model_provider",
    "resolve_oracle_model",
    "validate_oracle_model",
]
