from .BattleAMPPredictor import PredictorBattleAMP

__all__ = ["PredictorBattleAMP"]
"""BattleAMP oracle strategy and bundled reference implementation."""

from pep_compass.optimization.components.oracles.strategies.battleamp.oracle import BattleAMPBlackBox

__all__ = ["BattleAMPBlackBox"]
