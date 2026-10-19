"""Public lightweight APEX objective component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pep_compass.steps.oracles.strategies.black_box import BlackBoxOracle


@dataclass
class APEX(BlackBoxOracle):
    """Configure APEX evaluation; weights are loaded only during build.

    :param mic_aggregate: Mean or maximum of log2 MIC across selected pathogens.
    :param mic_bacteria: All pathogens or a non-empty list of zero-based indices.
    """

    implementation: ClassVar[str] = (
        "pep_compass.steps.oracles.strategies.apex.oracle.APEXBlackBox"
    )
    objective: ClassVar[str] = "apex"
    model: str = "default"
    models_directory: str | None = None
    device: str = "cpu"
    mic_aggregate: str = "mean"
    mic_bacteria: str | list[int] = "all"

    def __post_init__(self) -> None:
        """Reject invalid aggregation and pathogen selection without opening weights.

        :rtype: None
        """
        from pep_compass.steps.oracles.strategies.apex.artifacts import APEX_MODELS
        from pep_compass.steps.oracles.strategies.apex.validation import (
            validate_bacteria_indices,
        )

        super().__post_init__()
        if self.mic_aggregate not in {"mean", "max"}:
            raise ValueError("APEX mic_aggregate must be mean or max.")
        if self.model not in APEX_MODELS:
            raise ValueError("Unknown APEX model variant.")
        validate_bacteria_indices(
            self.mic_bacteria, pathogen_count=len(APEX_MODELS[self.model].pathogens)
        )
