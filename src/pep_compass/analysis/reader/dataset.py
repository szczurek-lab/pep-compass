"""Logical dataset table handles backed by lazy experiment selections."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pandas as pd

from pep_compass.analysis.reader.selection import ExperimentSelection
from pep_compass.optimization.saving.reader import AnalysisReader


class SelectionTableHandle:
    """Expose one selected logical table through the shared dataset contract."""

    def __init__(self, selection: ExperimentSelection, table: str) -> None:
        """Initialize this object with the declared configuration.

        :param selection: Strategy selecting admissible substitutions or candidate rows.
        :type selection: ExperimentSelection
        :param table: Table.
        :type table: str
        :rtype: None
        """
        self.selection = selection
        self.table = table

    @property
    def columns(self) -> tuple[str, ...]:
        """Return source columns available from the first existing table.

        :return: Result of this operation.
        :rtype: tuple[str, ...]
        """
        if not self.selection.runs:
            return ()
        with AnalysisReader(self.selection.runs[0].tracking_path) as reader:
            return reader.tables[self.table].columns

    def count_rows(self) -> int:
        """Count selected rows without materializing the table.

        :return: Result of this operation.
        :rtype: int
        """
        return self.selection.count_rows(self.table)

    def scan(
        self,
        columns: Sequence[str],
        *,
        chunk_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield projected table chunks with bounded memory.

        :param columns: Column names to project in the requested order.
        :type columns: Sequence[str]
        :param chunk_size: Maximum rows materialized in one scan chunk.
        :type chunk_size: int
        :return: Result of this operation.
        :rtype: Iterator[pd.DataFrame]
        """
        yield from self.selection.scan(
            self.table,
            columns=list(columns),
            chunk_size=chunk_size,
        )
