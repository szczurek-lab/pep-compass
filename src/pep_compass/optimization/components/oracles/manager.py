"""Oracle registry facade retained at the component-family boundary."""

from pep_compass.optimization.components.oracles.base import Oracle
from pep_compass.optimization.components.oracles.model_registry import (
    has_oracle_model_provider,
    validate_oracle_model,
)
from pep_compass.registry import Registry, component_catalog


class OracleManager:
    """Map configured oracle names to lazy factories used by ``PipelineBuilder``."""

    registry: Registry[Oracle] = Registry("oracle", expected_type=Oracle)

    @classmethod
    def register(
        cls, name: str, **kwargs
    ):
        """Register an oracle factory."""

        return cls.registry.register(name, **kwargs)

    @classmethod
    def build(cls, method: str, **parameters) -> Oracle:
        """Construct a registered oracle strategy."""
        return cls.registry.build(method, parameters)

    @classmethod
    def methods(cls) -> tuple[str, ...]:
        """Return registered method names in deterministic order.

        :return: Registered oracle method names.
        :rtype: tuple[str, ...]
        """
        return cls.registry.names()

    @classmethod
    def validate(cls, method: str, parameters) -> None:
        """Validate a declaration without loading an oracle model."""
        cls.registry.validate(method, parameters)
        if has_oracle_model_provider(method):
            validate_oracle_model(method, parameters.get("model", "default"))


component_catalog.register_family(
    "oracle", OracleManager.registry, validator=OracleManager.validate
)
