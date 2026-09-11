"""Parallel branch execution and deterministic output merging."""

from pep_compass.optimization.engine.operations.parallel.merge import build_merger
from pep_compass.optimization.engine.operations.parallel.operation import Parallel

__all__ = ["Parallel", "build_merger"]
