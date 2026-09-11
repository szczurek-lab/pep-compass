# Make EIPred a package and expose predictors
"""EIPred oracle strategy and bundled reference implementation."""

from pep_compass.optimization.components.oracles.strategies.eipred.oracle import EIPredBlackBox

__all__ = ["EIPredBlackBox"]

__all__ = ["PredictorEIPred", "EIPredPredictor"]
