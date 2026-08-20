"""Mutation-generator registry facade at the component-family boundary."""

from pep_compass.optimization.components.mutation_generators.base import MutationGenerator
from pep_compass.registry import Registry, component_catalog


class MutationGeneratorManager:
    """Map configured generator names to factories used by ``PipelineBuilder``."""

    registry: Registry[MutationGenerator] = Registry(
        "mutation generator", expected_type=MutationGenerator
    )

    @classmethod
    def register(
        cls,
        name: str,
        **kwargs,
    ):
        """Register a mutation generator factory."""
        kwargs.setdefault("services", {"autoencoder"})
        return cls.registry.register(name, **kwargs)

    @classmethod
    def build(cls, method: str, *, services=None, **parameters) -> MutationGenerator:
        """Construct a registered mutation generator."""
        return cls.registry.build(method, parameters, services=services)

    @classmethod
    def methods(cls) -> tuple[str, ...]:
        """Return registered mutation-generator method names."""
        return cls.registry.names()

    @classmethod
    def validate(cls, method: str, parameters) -> None:
        """Validate a declaration without constructing the generator."""
        cls.registry.validate(method, parameters)
        if method != "mutang":
            return
        from pep_compass.optimization.components.mutation_generators.strategies.mutang.registry import (
            combination_registry,
            direction_selection_registry,
            geometry_registry,
            mutation_selection_registry,
            scoring_registry,
        )

        strategies = parameters.get("strategies", {})
        if not isinstance(strategies, dict):
            raise ValueError("mutang.strategies must be a mapping.")
        registries = {
            "geometry": geometry_registry,
            "direction_selection": direction_selection_registry,
            "scoring": scoring_registry,
            "mutation_selection": mutation_selection_registry,
            "combination": combination_registry,
        }
        for name, registry in registries.items():
            declaration = strategies.get(name)
            if declaration is None:
                continue
            if not isinstance(declaration, dict) or not isinstance(
                declaration.get("method"), str
            ):
                raise ValueError(f"mutang.{name} must declare a string method.")
            nested_parameters = declaration.get("parameters", {})
            if not isinstance(nested_parameters, dict):
                raise ValueError(f"mutang.{name}.parameters must be a mapping.")
            registry.validate(declaration["method"], nested_parameters)


component_catalog.register_family(
    "mutation_generator",
    MutationGeneratorManager.registry,
    validator=MutationGeneratorManager.validate,
)
