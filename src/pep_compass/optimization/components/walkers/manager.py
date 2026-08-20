"""Walker registry facade retained at the component-family boundary."""

from pep_compass.optimization.components.walkers.base import Walker
from pep_compass.registry import Registry, component_catalog


class WalkerManager:
    """Map configured walker names to factories used by ``PipelineBuilder``."""

    registry: Registry[Walker] = Registry("walker", expected_type=Walker)

    @classmethod
    def register(
        cls, name: str, **kwargs
    ):
        """Register a walker factory under a configuration name."""
        kwargs.setdefault("services", {"autoencoder"})
        return cls.registry.register(name, **kwargs)

    @classmethod
    def build(cls, method: str, *, services=None, **parameters) -> Walker:
        """Construct a registered walker strategy."""
        return cls.registry.build(method, parameters, services=services)

    @classmethod
    def methods(cls) -> tuple[str, ...]:
        """Return registered walker method names."""
        return cls.registry.names()

    @classmethod
    def validate(cls, method: str, parameters) -> None:
        """Validate a declaration without constructing the walker."""
        cls.registry.validate(method, parameters)
        if method != "sorbes":
            return
        from pep_compass.optimization.components.walkers.strategies.sorbes.registry import (
            boundary_registry,
            directions_registry,
            geometry_registry,
            position_update_registry,
            scaling_registry,
        )

        registries = {
            "geometry": geometry_registry,
            "directions": directions_registry,
            "scaling": scaling_registry,
            "position_update": position_update_registry,
            "boundary": boundary_registry,
        }
        for name, registry in registries.items():
            declaration = parameters.get(name)
            if declaration is None:
                continue
            if not isinstance(declaration, dict) or not isinstance(
                declaration.get("method"), str
            ):
                raise ValueError(f"sorbes.{name} must declare a string method.")
            nested_parameters = declaration.get("parameters", {})
            if not isinstance(nested_parameters, dict):
                raise ValueError(f"sorbes.{name}.parameters must be a mapping.")
            registry.validate(declaration["method"], nested_parameters)


component_catalog.register_family(
    "walker", WalkerManager.registry, validator=WalkerManager.validate
)
