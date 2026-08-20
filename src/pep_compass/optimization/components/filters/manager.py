"""Filter registry facade retained at the component-family boundary."""

from pep_compass.optimization.components.filters.base import Filter
from pep_compass.registry import Registry, component_catalog


class FilterManager:
    """Map configured filter names to factories used by ``PipelineBuilder``."""

    registry: Registry[Filter] = Registry("filter", expected_type=Filter)

    @classmethod
    def register(
        cls, name: str, **kwargs
    ):
        """Register a filter factory."""
        kwargs.setdefault("services", {"autoencoder"})
        return cls.registry.register(name, **kwargs)

    @classmethod
    def build(cls, method: str, *, services=None, **parameters) -> Filter:
        """Construct a registered filter strategy."""
        return cls.registry.build(method, parameters, services=services)

    @classmethod
    def methods(cls) -> tuple[str, ...]:
        """Return registered filter method names."""
        return cls.registry.names()

    @classmethod
    def validate(cls, method: str, parameters) -> None:
        """Validate a strategy declaration without constructing the filter."""
        cls.registry.validate(method, parameters)
        if method != "ranked":
            return
        from pep_compass.optimization.components.filters.ranked.registry import (
            scoring_registry,
            selection_registry,
        )

        cls._validate_nested("scoring", parameters["scoring"], scoring_registry)
        cls._validate_nested("selection", parameters["selection"], selection_registry)

    @staticmethod
    def _validate_nested(name: str, declaration, registry) -> None:
        """Validate one ``{method, parameters}`` nested declaration."""
        if not isinstance(declaration, dict) or not isinstance(
            declaration.get("method"), str
        ):
            raise ValueError(f"ranked.{name} must declare a string method.")
        parameters = declaration.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"ranked.{name}.parameters must be a mapping.")
        registry.validate(declaration["method"], parameters)


component_catalog.register_family(
    "filter", FilterManager.registry, validator=FilterManager.validate
)
