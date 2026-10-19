"""Hydra configuration groups derived from direct import targets and YAML groups."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from hydra import compose, initialize_config_dir, initialize_config_module

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omegaconf import DictConfig


@dataclass(frozen=True)
class OrchestrationSettings:
    """Independent-job identity, output root and failure policy."""

    name: str = "experiment"
    output_dir: str = "results"
    skip_completed: bool = False
    continue_on_error: bool = False

    def __post_init__(self) -> None:
        """Post init.

        :rtype: None
        """
        if (
            not isinstance(self.name, str)
            or not self.name
            or Path(self.name).name != self.name
            or self.name in {".", ".."}
        ):
            raise ValueError(
                "orchestrator.name must be a single non-empty path component."
            )
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("orchestrator.output_dir must be a non-empty path.")
        for name in ("skip_completed", "continue_on_error"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"orchestrator.{name} must be a boolean.")


def load_plugins(modules: Sequence[str]) -> None:
    """Import explicitly requested explicit modules before composing configs.

    :param modules: Python module names supplied by the application owner.
    :raises ValueError: If names are not a sequence of non-empty strings.

    :type modules: Sequence[str]
    :rtype: None
    """
    if isinstance(modules, (str, bytes)) or any(
        not isinstance(name, str) or not name for name in modules
    ):
        raise ValueError("plugins must contain Python module names.")
    for name in modules:
        import_module(name)


def compose_configuration(
    overrides: Sequence[str] = (),
    *,
    config_dir: str | Path | None = None,
    config_name: str = "config",
    plugins: Sequence[str] = (),
) -> DictConfig:
    """Compose Hydra defaults, interpolation and overrides from Python.

    :param overrides: Hydra override expressions.
    :param config_dir: Optional custom configuration directory.
    :param config_name: Root YAML name without its extension.
    :param plugins: Explicit application modules included in the frozen source identity.
    :return: Composed OmegaConf DictConfig; no models are constructed.

    :type overrides: Sequence[str]
    :type config_dir: str | Path | None
    :type config_name: str
    :type plugins: Sequence[str]
    :rtype: DictConfig
    """
    load_plugins(plugins)
    context = (
        initialize_config_module(
            version_base="1.3", config_module="pep_compass.orchestrator.conf"
        )
        if config_dir is None
        else initialize_config_dir(
            version_base="1.3", config_dir=str(Path(config_dir).resolve())
        )
    )
    with context:
        configuration = compose(config_name=config_name, overrides=list(overrides))
    from omegaconf import open_dict

    with open_dict(configuration):
        configuration.plugins = list(
            dict.fromkeys([*configuration.get("plugins", []), *plugins])
        )
    return configuration


def plugin_fingerprints(modules: Sequence[str]) -> dict[str, str]:
    """Hash explicitly imported plugin module files for frozen-plan identity.

    :param modules: Importable explicit modules.
    :return: Mapping from module name to its source/artifact SHA-256.
    :raises ValueError: If a plugin has no inspectable file.

    :type modules: Sequence[str]
    :rtype: dict[str, str]
    """
    import hashlib
    import inspect

    result = {}
    for name in modules:
        module = import_module(name)
        source = inspect.getsourcefile(module) or getattr(module, "__file__", None)
        if source is None:
            raise ValueError(f"Plugin {name!r} has no inspectable source file.")
        result[name] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return result
