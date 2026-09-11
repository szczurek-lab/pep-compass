"""Contract tests applied automatically to registered components."""

import subprocess
import sys

import pep_compass.optimization.components.filters.registry  # noqa: F401
import pep_compass.optimization.components.mutation_generators.strategies  # noqa: F401
import pep_compass.optimization.components.walkers.strategies  # noqa: F401
import pep_compass.optimization.components.oracles.strategies  # noqa: F401
import pytest

from pep_compass.optimization.components.filters import FilterManager
from pep_compass.optimization.components.mutation_generators import (
    MutationGeneratorManager,
)
from pep_compass.optimization.components.oracles import OracleManager
from pep_compass.optimization.components.walkers import WalkerManager


@pytest.mark.parametrize(
    "registry",
    (
        WalkerManager.registry,
        MutationGeneratorManager.registry,
        FilterManager.registry,
        OracleManager.registry,
    ),
)
def test_registered_component_factories_have_stable_non_empty_names(registry) -> None:
    """Every component family must expose callable factories for validation."""
    assert registry.names()
    assert all(isinstance(name, str) and name for name in registry.names())
    assert all(callable(factory) for factory in registry.entries.values())


def test_oracle_registration_does_not_import_model_implementations() -> None:
    """Discovering oracle names must not initialize optional model stacks."""
    script = """
import sys
import pep_compass.optimization.components.oracles.strategies

implementation_modules = {
    'pep_compass.optimization.components.oracles.strategies.apex.oracle',
    'pep_compass.optimization.components.oracles.strategies.apex_original.oracle',
    'pep_compass.optimization.components.oracles.strategies.battleamp.oracle',
    'pep_compass.optimization.components.oracles.strategies.eipred.oracle',
    'pep_compass.optimization.components.oracles.strategies.hydrophobicity.oracle',
    'pep_compass.optimization.components.oracles.strategies.mbc_attention.oracle',
    'pep_compass.optimization.components.oracles.strategies.toxipep.oracle',
}
assert implementation_modules.isdisjoint(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
    assert OracleManager.methods() == (
        "apex",
        "apex_original",
        "battleamp",
        "eipred",
        "hydrophobicity",
        "mbc_attention",
        "toxipep",
    )


def test_nested_strategy_registries_reject_unknown_methods() -> None:
    """Nested SORBES, MUTANG, and ranked methods must be validated explicitly."""
    with pytest.raises(ValueError, match="Unknown SORBES geometry strategy"):
        WalkerManager.validate(
            "sorbes",
            {"geometry": {"method": "missing", "parameters": {}}},
        )
    with pytest.raises(ValueError, match="Unknown MUTANG geometry strategy"):
        MutationGeneratorManager.validate(
            "mutang",
            {"strategies": {"geometry": {"method": "missing"}}},
        )
    with pytest.raises(ValueError, match="Unknown ranked-filter scoring strategy"):
        FilterManager.validate(
            "ranked",
            {
                "scoring": {"method": "missing", "parameters": {}},
                "selection": {
                    "method": "threshold",
                    "parameters": {"threshold": 0.5},
                },
            },
        )


def test_oracle_registry_rejects_unknown_named_model() -> None:
    """Oracle configuration validation must reject unknown model variants."""
    with pytest.raises(ValueError, match="Unknown model oracle.battleamp/missing"):
        OracleManager.validate("battleamp", {"model": "missing"})
