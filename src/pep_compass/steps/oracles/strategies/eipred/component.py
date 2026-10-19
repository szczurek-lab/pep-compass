"""Public lightweight EIPred objective component."""

from dataclasses import dataclass
from typing import ClassVar

from pep_compass.steps.oracles.strategies.black_box import BlackBoxOracle


@dataclass
class EIPred(BlackBoxOracle):
    """Configure EIPred evaluation; model initialization occurs during build."""

    implementation: ClassVar[str] = (
        "pep_compass.steps.oracles.strategies.eipred.oracle.EIPredBlackBox"
    )
    objective: ClassVar[str] = "eipred"
    model: str = "default"
    models_directory: str | None = None
    mic_aggregate: str = "mean"
