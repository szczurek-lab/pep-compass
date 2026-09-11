"""Oracle component contracts, registry and implementations."""

from pep_compass.optimization.components.oracles.base import Oracle
from pep_compass.optimization.components.oracles.manager import OracleManager

__all__ = ["Oracle", "OracleManager"]
