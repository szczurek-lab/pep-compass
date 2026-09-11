"""Base adapter for one versioned experiment data table."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class DataSchema:
    """Validate and lazily scan one logical tracker table.

    :param name: Logical table name exposed by the reader.
    :param filename: CSV filename in a tracking directory.
    :param required_columns: Columns required by the canonical schema.
    """

    name: str
    filename: str
    required_columns: frozenset[str]

    def inspect(self, path: Path) -> list[str]:
        """Return available columns and validate the canonical minimum schema.

        :param path: CSV file to inspect.
        :return: Source column names in file order.
        :raises ValueError: If required columns are missing.
        """
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        missing = self.required_columns - set(columns)
        if missing:
            raise ValueError(
                f"Invalid {self.name} schema in {path}; missing {sorted(missing)}"
            )
        return columns

    def scan(
        self,
        path: Path,
        columns: list[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        chunk_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield projected and filtered CSV chunks with bounded memory.

        :param path: CSV file to scan.
        :param columns: Canonical columns returned to the caller.
        :param filters: Equality or membership filters applied per chunk.
        :param chunk_size: Maximum rows loaded from the source at once.
        :return: Iterator of normalized data frames.
        :raises KeyError: If a requested or filtered column is unavailable.
        :raises ValueError: If ``chunk_size`` is not positive.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        available = self.inspect(path)
        filters = filters or {}
        needed = list(dict.fromkeys([*(columns or []), *filters]))
        missing = set(needed) - set(available)
        if missing:
            raise KeyError(f"Missing columns in {path}: {sorted(missing)}")
        usecols = needed or None
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
            for column, accepted in filters.items():
                values = accepted if isinstance(accepted, (list, tuple, set)) else [accepted]
                chunk = chunk[chunk[column].isin(values)]
            if chunk.empty:
                continue
            if columns is not None:
                chunk = chunk[columns]
            yield self.normalize(chunk)

    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a canonical frame; subclasses may normalize legacy values."""
        return frame
