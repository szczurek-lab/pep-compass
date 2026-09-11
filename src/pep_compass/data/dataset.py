"""Logical, lazily accessible representation of loaded experiment results."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class TableHandle(Protocol):
    """Provide bounded, column-oriented access to one persisted result table."""

    @property
    def columns(self) -> tuple[str, ...]:
        """Return columns available from the table."""

    def count_rows(self) -> int:
        """Return the number of rows visible through this handle."""

    def scan(
        self,
        columns: Sequence[str],
        *,
        chunk_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield bounded table chunks containing selected columns."""


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Identify one independently executable experiment run."""

    run_id: str
    experiment: str
    variant_id: str


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Describe one loaded run without materializing its result tables."""

    identity: RunIdentity
    root: Path
    configuration: Mapping[str, Any] = field(compare=False, repr=False)
    metadata: Mapping[str, Any] = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class ExperimentDataset:
    """Expose discovered runs and lazy tables through a versioned contract."""

    root: Path
    schema_version: str
    runs: tuple[RunRecord, ...]
    tables: Mapping[str, TableHandle] = field(compare=False, repr=False)
