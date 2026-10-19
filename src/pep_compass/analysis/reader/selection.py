"""Immutable lazy selections over discovered experiment runs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import pandas as pd

from pep_compass.analysis.reader.entities import ExperimentRun
from pep_compass.optimization.saving.reader import AnalysisReader

if TYPE_CHECKING:
    from pep_compass.analysis.reader.reader import ExperimentReader


@dataclass(frozen=True, slots=True)
class ExperimentSelection:
    """Describe selected runs and deferred row filters.

    :param reader: Parent reader owning discovery metadata and metric cache.
    :param runs: Selected run entities.
    :param row_filters: Table-row filters applied only when data is scanned.
    """

    reader: "ExperimentReader"
    runs: tuple[ExperimentRun, ...]
    row_filters: Mapping[str, Any]

    def rows(self, **filters: Any) -> "ExperimentSelection":
        """Return a copy with additional deferred table-row filters.

        :param filters: Filters.
        :type filters: Any
        :return: Result of this operation.
        :rtype: ExperimentSelection
        """
        return replace(self, row_filters={**self.row_filters, **filters})

    def select(
        self,
        experiments: list[str] | None = None,
        methods: list[str] | None = None,
        grid_ids: list[str] | None = None,
        peptides: list[str] | None = None,
        seeds: list[int] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> "ExperimentSelection":
        """Further restrict runs using manifest and resolved-config metadata.

        :param experiments: Experiments.
        :type experiments: list[str] | None
        :param methods: Methods.
        :type methods: list[str] | None
        :param grid_ids: Grid ids.
        :type grid_ids: list[str] | None
        :param peptides: Peptide strings to encode.
        :type peptides: list[str] | None
        :param seeds: Seeds.
        :type seeds: list[int] | None
        :param parameters: Named configuration values consumed by this operation.
        :type parameters: Mapping[str, Any] | None
        :return: Result of this operation.
        :rtype: ExperimentSelection
        """
        selectors = {
            "experiment": experiments,
            "method": methods,
            "grid_id": grid_ids,
            "name": peptides,
            "seed": seeds,
        }
        selected = []
        for run in self.runs:
            metadata = run.metadata
            if any(
                values is not None and metadata.get(key) not in values
                for key, values in selectors.items()
            ):
                continue
            if any(
                metadata.get(key) not in (value if isinstance(value, list) else [value])
                for key, value in (parameters or {}).items()
            ):
                continue
            selected.append(run)
        return replace(self, runs=tuple(selected))

    def scan(
        self,
        table: str,
        columns: list[str] | None = None,
        chunk_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield selected table rows with run metadata attached lazily.

        :param table: Logical table name registered in ``data_schemas``.
        :param columns: Source columns requested by the analysis.
        :param chunk_size: Maximum source rows read at once.
        :return: Iterator of canonical chunks enriched with run metadata.

        :type table: str
        :type columns: list[str] | None
        :type chunk_size: int
        :rtype: Iterator[pd.DataFrame]
        """
        for run in self.runs:
            with AnalysisReader(run.tracking_path) as source:
                handle = source.tables[table]
                available = set(handle.columns)
                requested = (
                    list(columns) if columns is not None else list(handle.columns)
                )
                metadata = [name for name in requested if name not in available]
                missing = set(metadata) - run.metadata.keys()
                if missing:
                    raise KeyError(f"Missing columns: {sorted(missing)}.")
                filters = {
                    key: value
                    for key, value in self.row_filters.items()
                    if key in available
                }
                projected = [name for name in requested if name in available]
                required = list(dict.fromkeys([*projected, *filters])) or [
                    handle.columns[0]
                ]
                for chunk in handle.scan(required, chunk_size=chunk_size):
                    for key, value in filters.items():
                        chunk = chunk[
                            chunk[key].isin(value)
                            if isinstance(value, (list, tuple, set))
                            else chunk[key].eq(value)
                        ]
                    for key in metadata:
                        chunk[key] = run.metadata[key]
                    yield chunk[requested]

    def collect(
        self,
        table: str,
        columns: list[str] | None = None,
        chunk_size: int = 100_000,
    ) -> pd.DataFrame:
        """Materialize a selected table when the caller accepts its memory cost.

        :param table: Table.
        :type table: str
        :param columns: Column names to project in the requested order.
        :type columns: list[str] | None
        :param chunk_size: Maximum rows materialized in one scan chunk.
        :type chunk_size: int
        :return: Result of this operation.
        :rtype: pd.DataFrame
        """
        chunks = list(self.scan(table, columns=columns, chunk_size=chunk_size))
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def count_rows(self, table: str, chunk_size: int = 500_000) -> int:
        """Count selected rows without materializing a tracker table.

        A newline count is used when no deferred filter applies to the table.
        Filtered selections are counted through projected chunks containing
        only the filter columns.

        :param table: Logical table name registered in ``data_schemas``.
        :param chunk_size: Rows per chunk when deferred filters require parsing.
        :return: Number of selected data rows across all runs.

        :type table: str
        :type chunk_size: int
        :rtype: int
        """
        if not self.row_filters:
            total = 0
            for run in self.runs:
                with AnalysisReader(run.tracking_path) as reader:
                    total += reader.tables[table].count_rows()
            return total
        return sum(len(chunk) for chunk in self.scan(table, chunk_size=chunk_size))

    def paths(self, table: str) -> tuple[object, ...]:
        """Return canonical SQLite bundle paths containing the logical table.

        :param table: Table.
        :type table: str
        :return: Result of this operation.
        :rtype: tuple[object, ...]
        """
        paths = []
        for run in self.runs:
            with AnalysisReader(run.tracking_path) as reader:
                if table in reader.tables:
                    paths.append(run.tracking_path / "results.sqlite")
        return tuple(paths)

    def specification(self) -> dict[str, Any]:
        """Return stable selection metadata used as an analysis-cache key.

        :return: Result of this operation.
        :rtype: dict[str, Any]
        """
        input_files = []
        for run in self.runs:
            for path in sorted(run.tracking_path.glob("*")):
                if not path.is_file():
                    continue
                stat = path.stat()
                input_files.append(
                    {
                        "path": str(path),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
        return {
            "run_ids": sorted(run.run_id for run in self.runs),
            "tracking_paths": sorted(str(run.tracking_path) for run in self.runs),
            "row_filters": dict(self.row_filters),
            "input_files": input_files,
        }
