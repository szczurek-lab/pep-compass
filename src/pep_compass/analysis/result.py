"""Common numerical result returned by experiment analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Bundle a numerical table with interpretation and diagnostics."""

    data: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
