"""Public lightweight ToxiPep objective component."""

from dataclasses import dataclass
from typing import ClassVar

from pep_compass.steps.oracles.strategies.black_box import BlackBoxOracle


@dataclass
class ToxiPep(BlackBoxOracle):
    """Configure ToxiPep evaluation; model initialization occurs during build."""

    implementation: ClassVar[str] = (
        "pep_compass.steps.oracles.strategies.toxipep.oracle.ToxiPepBlackBox"
    )
    objective: ClassVar[str] = "toxipep"
    model: str = "default"
    models_directory: str | None = None
    device: str = "cpu"
