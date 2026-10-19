"""Public lightweight BattleAMP objective component."""

from dataclasses import dataclass
from typing import ClassVar

from pep_compass.steps.oracles.strategies.black_box import BlackBoxOracle


@dataclass
class BattleAMP(BlackBoxOracle):
    """Configure BattleAMP evaluation; model initialization occurs during build."""

    implementation: ClassVar[str] = (
        "pep_compass.steps.oracles.strategies.battleamp.oracle.BattleAMPBlackBox"
    )
    objective: ClassVar[str] = "battleamp"
    model: str = "default"
    models_directory: str | None = None
    device: str = "cpu"
