"""Portable constructor snapshots and direct import-based object construction."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from math import isfinite
from pathlib import PurePath
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from typing import Self

    from pep_compass.autoencoder.base import Autoencoder


ConfigValue: TypeAlias = (
    None | bool | int | float | str | list["ConfigValue"] | dict[str, "ConfigValue"]
)
FrozenConfigValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["FrozenConfigValue", ...]
    | Mapping[str, "FrozenConfigValue"]
)


def target_path(value: object) -> str:
    """Return the importable defining path of an object or class.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: str
    """
    kind = value if isinstance(value, type) else type(value)
    if "<locals>" in kind.__qualname__:
        raise ValueError("Persisted components must be defined in importable modules.")
    return kind.__module__ + "." + kind.__qualname__


def locate(path: str) -> object:
    """Import one explicitly selected target, without a discovery catalog.

    :param path: Input or output path used by this operation.
    :type path: str
    :return: Result of this operation.
    :rtype: object
    """
    if not isinstance(path, str) or "." not in path:
        raise ValueError("A target must be a qualified import path.")
    module, _, name = path.rpartition(".")
    return getattr(import_module(module), name)


def plain(value: object) -> ConfigValue:
    """Snapshot constructor values without retaining live computational state.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: ConfigValue
    """
    if isinstance(value, Configurable):
        return value.to_config()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Configuration mappings require string keys.")
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, PurePath):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    if hasattr(value, "detach"):
        return plain(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    raise TypeError(f"Non-portable constructor parameter: {type(value).__name__}.")


def freeze(value: object) -> FrozenConfigValue:
    """Return recursively immutable configuration containers.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: FrozenConfigValue
    """
    value = plain(value)
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def construct(value: object) -> object:
    """Instantiate explicit constructor snapshots, recursively and without weights.

    :param value: Plain values or mappings containing a qualified ``_target_``.
    :return: Fresh constructor objects; no global registrations are consulted.

    :type value: object
    :rtype: object
    """
    if isinstance(value, Mapping):
        if "_target_" in value:
            target = locate(value["_target_"])
            parameters = {
                key: construct(item) for key, item in value.items() if key != "_target_"
            }
            return target(**parameters)
        return {key: construct(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [construct(item) for item in value]
    return value


class Configurable:
    """Expose one constructor as the source of configuration and defaults.

    Concrete constructors call ``_configure`` with their public arguments.
    This base owns copying, serialization and nested resource initialization;
    it does not select implementations or maintain a catalog.
    """

    def __init__(self) -> None:
        """Initialize this object with the declared configuration.

        :rtype: None
        """
        self._configure({})

    def _configure(self, parameters: Mapping[str, object]) -> None:
        """Configure.

        :param parameters: Named configuration values consumed by this operation.
        :type parameters: Mapping[str, object]
        :rtype: None
        """
        self._configuration = freeze(parameters)

    @property
    def configuration(self) -> dict[str, ConfigValue]:
        """Return an independent plain snapshot of all resolved constructor values.

        :return: Result of this operation.
        :rtype: dict[str, ConfigValue]
        """
        return plain(getattr(self, "_configuration", {}))

    def to_config(self) -> dict[str, ConfigValue]:
        """Return a Hydra-compatible direct constructor declaration.

        :return: Result of this operation.
        :rtype: dict[str, ConfigValue]
        """
        return {"_target_": target_path(self), **self.configuration}

    def initialize(self, autoencoder: Autoencoder | None = None) -> Self:
        """Prepare declared nested resources only when compiling a pipeline.

        :param autoencoder: Shared initialized autoencoder.
        :return: This prepared object.

        :type autoencoder: Autoencoder | None
        :rtype: Self
        """
        if hasattr(self, "autoencoder") and getattr(self, "autoencoder") is None:
            self.autoencoder = autoencoder
        for value in vars(self).values():
            if isinstance(value, Configurable):
                value.initialize(autoencoder)
            elif isinstance(value, (list, tuple, dict)):
                for child in value.values() if isinstance(value, dict) else value:
                    if isinstance(child, Configurable):
                        child.initialize(autoencoder)
        prepare = getattr(self, "_initialize", None)
        if prepare is not None:
            prepare(autoencoder)
        return self
