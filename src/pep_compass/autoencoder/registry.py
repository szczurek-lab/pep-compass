"""Registry of autoencoder methods and named model variants."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pep_compass.autoencoder.base import Autoencoder
from pep_compass.registry import (
    ModelArtifact,
    ModelDescriptor,
    Registry,
    model_catalog,
)


AutoencoderFactoryCallable = Callable[..., Autoencoder]


@dataclass(frozen=True, slots=True)
class AutoencoderModelDescriptor:
    """Describe a named model variant supported by one autoencoder method.

    :param name: Public model name used in configuration.
    :param parameters: Loader parameters associated with the model variant.
    """

    name: str
    parameters: dict[str, Any]
    directory: str | None = None
    artifacts: tuple[ModelArtifact, ...] = ()
    download_script: str | None = None


class AutoencoderRegistry:
    """Store autoencoder factories and named model descriptors."""

    _method_registry: Registry[Autoencoder] = Registry(
        "autoencoder", expected_type=Autoencoder
    )

    @classmethod
    def register_method(
        cls,
        name: str,
    ) -> Callable[[AutoencoderFactoryCallable], AutoencoderFactoryCallable]:
        """Return a decorator registering one autoencoder implementation."""
        return cls._method_registry.register(name)

    @classmethod
    def register_model(
        cls,
        method: str,
        descriptor: AutoencoderModelDescriptor,
    ) -> None:
        """Register a named model variant for an autoencoder method."""
        model_catalog.register(
            ModelDescriptor(
                provider=f"autoencoder.{method}",
                name=descriptor.name,
                directory=descriptor.directory or descriptor.name,
                artifacts=descriptor.artifacts,
                parameters=dict(descriptor.parameters),
                download_script=descriptor.download_script,
            )
        )

    @classmethod
    def method(cls, name: str) -> AutoencoderFactoryCallable:
        """Return a registered autoencoder factory."""
        return cls._method_registry.factory(name)

    @classmethod
    def model(cls, method: str, name: str) -> ModelDescriptor:
        """Return a named model descriptor for a method."""
        try:
            return model_catalog.descriptor(f"autoencoder.{method}", name)
        except ValueError as error:
            raise ValueError(
                f"Unknown autoencoder model: {method}/{name}. "
                f"Available: {list(cls.models(method))}."
            ) from error

    @classmethod
    def methods(cls) -> tuple[str, ...]:
        """Return registered method names in deterministic order."""
        return cls._method_registry.names()

    @classmethod
    def models(cls, method: str) -> tuple[str, ...]:
        """Return named model variants registered for a method."""
        return model_catalog.names(f"autoencoder.{method}")
