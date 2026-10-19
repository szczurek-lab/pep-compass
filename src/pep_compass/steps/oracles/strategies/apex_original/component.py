"""Public lightweight APEXOriginal objective component."""

from dataclasses import dataclass
from typing import ClassVar

from pep_compass.steps.oracles.strategies.apex.component import APEX


@dataclass
class APEXOriginal(APEX):
    """Configure APEXOriginal evaluation; model initialization occurs during build."""

    implementation: ClassVar[str] = (
        "pep_compass.steps.oracles.strategies.apex_original.oracle.APEXBlackBox"
    )
    objective: ClassVar[str] = "apex_original"
