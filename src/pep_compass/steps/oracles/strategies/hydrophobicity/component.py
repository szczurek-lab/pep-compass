"""Public lightweight Hydrophobicity objective component."""

from dataclasses import dataclass
from typing import ClassVar

from pep_compass.steps.oracles.strategies.black_box import BlackBoxOracle


@dataclass
class Hydrophobicity(BlackBoxOracle):
    """Evaluate mean amino-acid hydrophobicity with larger scores preferred.

    :param scale: modlAMP scale name; defaults to eisenberg.
    """

    implementation: ClassVar[str] = (
        "pep_compass.steps.oracles.strategies.hydrophobicity.oracle.HydrophobicityBlackBox"
    )
    objective: ClassVar[str] = "hydrophobicity"
    scale: str = "eisenberg"
