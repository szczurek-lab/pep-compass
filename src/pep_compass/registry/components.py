"""Catalog of component families accepted by the pipeline configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pep_compass.registry.core import Registry


@dataclass(frozen=True, slots=True)
class ComponentFamily:
    """Bind one strategy registry to optional family-level validation."""

    registry: Registry[Any]
    validator: Callable[[str, Mapping[str, Any]], None] | None = None


class ComponentCatalog:
    """Map pipeline component kinds to their namespaced strategy registries."""

    def __init__(self) -> None:
        self._families: dict[str, ComponentFamily] = {}

    def register_family(
        self,
        name: str,
        registry: Registry[Any],
        *,
        validator: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        """Register one unique pipeline component family."""
        if not name:
            raise ValueError("Component family name cannot be empty.")
        if name in self._families:
            raise ValueError(f"Duplicate component family registration: {name}.")
        self._families[name] = ComponentFamily(registry, validator)

    def names(self) -> tuple[str, ...]:
        """Return registered component-family names."""
        return tuple(sorted(self._families))

    def family(self, name: str) -> ComponentFamily:
        """Return a registered component-family registry."""
        try:
            return self._families[name]
        except KeyError as error:
            raise ValueError(
                f"Unknown component family: {name!r}. "
                f"Available: {list(self.names())}."
            ) from error

    def validate(
        self,
        family: str,
        method: str,
        parameters: Mapping[str, Any],
    ) -> None:
        """Validate one component strategy declaration."""
        descriptor = self.family(family)
        if descriptor.validator is not None:
            descriptor.validator(method, parameters)
        else:
            descriptor.registry.validate(method, parameters)

    def build(
        self,
        family: str,
        method: str,
        parameters: Mapping[str, Any],
        *,
        services: Mapping[str, Any] | None = None,
    ) -> Any:
        """Validate and build one component through its registered family."""
        descriptor = self.family(family)
        if descriptor.validator is not None:
            descriptor.validator(method, parameters)
        return descriptor.registry.build(
            method, parameters, services=services
        )


component_catalog = ComponentCatalog()
