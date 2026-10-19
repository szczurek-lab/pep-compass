"""Public lightweight MBCAttention objective component."""

from dataclasses import dataclass
from typing import ClassVar

from pep_compass.steps.oracles.strategies.black_box import BlackBoxOracle


@dataclass
class MBCAttention(BlackBoxOracle):
    """Configure MBCAttention evaluation; model initialization occurs during build."""

    implementation: ClassVar[str] = (
        "pep_compass.steps.oracles.strategies.mbc_attention.oracle.MBCAttentionBlackBox"
    )
    objective: ClassVar[str] = "mbc_attention"
    model: str = "default"
    models_directory: str | None = None
    device: str = "cpu"
