"""Persistent SQLite cache for expensive metrics and completed analyses."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible analysis metadata."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MetricsStore:
    """Cache sequence metrics and reproducible aggregate analysis results."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the SQLite schema.

        :param path: SQLite file stored beside the opened experiment collection.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metric_values (
                    sequence_hash TEXT NOT NULL,
                    sequence TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_version TEXT NOT NULL,
                    configuration_hash TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    PRIMARY KEY (
                        sequence_hash, metric_name, metric_version,
                        configuration_hash
                    )
                );
                CREATE TABLE IF NOT EXISTS analysis_results (
                    analysis_name TEXT NOT NULL,
                    analysis_version TEXT NOT NULL,
                    selection_hash TEXT NOT NULL,
                    parameters_hash TEXT NOT NULL,
                    selection_json TEXT NOT NULL DEFAULT '{}',
                    parameters_json TEXT NOT NULL DEFAULT '{}',
                    frame_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    complete INTEGER NOT NULL,
                    PRIMARY KEY (
                        analysis_name, analysis_version, selection_hash,
                        parameters_hash
                    )
                );
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(analysis_results)")
            }
            if "selection_json" not in columns:
                connection.execute(
                    "ALTER TABLE analysis_results ADD COLUMN "
                    "selection_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "parameters_json" not in columns:
                connection.execute(
                    "ALTER TABLE analysis_results ADD COLUMN "
                    "parameters_json TEXT NOT NULL DEFAULT '{}'"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def put_metrics(
        self,
        metric_name: str,
        values: Mapping[str, Any],
        metric_version: str,
        configuration: Mapping[str, Any] | None = None,
    ) -> None:
        """Insert or update sequence metrics in one transaction.

        :param metric_name: Metric identifier, for example ``apex`` or ``pogs``.
        :param values: Sequence-to-value mapping; values must be JSON-compatible.
        :param metric_version: Model or implementation fingerprint.
        :param configuration: Metric parameters affecting reproducibility.
        """
        configuration_hash = stable_hash(configuration or {})
        rows = [
            (
                stable_hash(sequence),
                sequence,
                metric_name,
                metric_version,
                configuration_hash,
                json.dumps(value, separators=(",", ":"), default=float),
            )
            for sequence, value in values.items()
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO metric_values VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_metrics(
        self,
        metric_name: str,
        sequences: Iterable[str],
        metric_version: str,
        configuration: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return cached values for the requested sequences.

        :param metric_name: Metric identifier.
        :param sequences: Candidate sequences to retrieve.
        :param metric_version: Model or implementation fingerprint.
        :param configuration: Metric parameters affecting reproducibility.
        :return: Mapping containing only cached sequences.
        """
        sequence_list = list(dict.fromkeys(sequences))
        if not sequence_list:
            return {}
        configuration_hash = stable_hash(configuration or {})
        result: dict[str, Any] = {}
        with self._connect() as connection:
            for start in range(0, len(sequence_list), 500):
                batch = sequence_list[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                query = f"""
                    SELECT sequence, value_json FROM metric_values
                    WHERE metric_name = ? AND metric_version = ?
                      AND configuration_hash = ?
                      AND sequence IN ({placeholders})
                """
                parameters = [
                    metric_name,
                    metric_version,
                    configuration_hash,
                    *batch,
                ]
                for sequence, value_json in connection.execute(query, parameters):
                    result[sequence] = json.loads(value_json)
        return result

    def get_or_compute_metrics(
        self,
        metric_name: str,
        sequences: Iterable[str],
        evaluator: Callable[[list[str]], Mapping[str, Any] | Sequence[Any]],
        metric_version: str,
        configuration: Mapping[str, Any] | None = None,
        batch_size: int = 256,
    ) -> dict[str, Any]:
        """Evaluate only missing sequence metrics and persist every batch.

        :param metric_name: Metric identifier, for example ``apex``.
        :param sequences: Sequences requiring metric values.
        :param evaluator: Batch callable returning either a sequence-aligned
            value list or an explicit sequence-to-value mapping.
        :param metric_version: Model or implementation fingerprint.
        :param configuration: Metric parameters affecting reproducibility.
        :param batch_size: Maximum uncached sequences evaluated at once.
        :return: Complete mapping for all requested sequences.
        :raises ValueError: If evaluator output length is inconsistent.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        requested = list(dict.fromkeys(sequences))
        cached = self.get_metrics(
            metric_name, requested, metric_version, configuration
        )
        missing = [sequence for sequence in requested if sequence not in cached]
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            evaluated = evaluator(batch)
            if isinstance(evaluated, Mapping):
                values = dict(evaluated)
            else:
                if len(evaluated) != len(batch):
                    raise ValueError("Evaluator output length does not match batch")
                values = dict(zip(batch, evaluated))
            absent = set(batch) - set(values)
            if absent:
                raise ValueError(f"Evaluator omitted sequences: {sorted(absent)}")
            self.put_metrics(
                metric_name,
                values,
                metric_version,
                configuration,
            )
            cached.update(values)
        return {sequence: cached[sequence] for sequence in requested}

    def put_analysis(
        self,
        analysis_name: str,
        frame: pd.DataFrame,
        metadata: Mapping[str, Any],
        selection: Mapping[str, Any],
        parameters: Mapping[str, Any],
        analysis_version: str = "1",
        complete: bool = True,
    ) -> None:
        """Store one exact aggregate result for later reconstruction.

        :param analysis_name: Stable analysis identifier.
        :param frame: Materialized numerical result.
        :param metadata: Result interpretation and diagnostics.
        :param selection: Exact experiment selection specification.
        :param parameters: Exact numerical analysis parameters.
        :param analysis_version: Implementation version.
        :param complete: Whether raw inputs are no longer required to reproduce
            this exact result.
        """
        row = (
            analysis_name,
            analysis_version,
            stable_hash(selection),
            stable_hash(parameters),
            json.dumps(selection, sort_keys=True, default=str),
            json.dumps(parameters, sort_keys=True, default=str),
            frame.to_json(orient="table", index=False),
            json.dumps(metadata, sort_keys=True, default=str),
            int(complete),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO analysis_results (
                    analysis_name, analysis_version, selection_hash,
                    parameters_hash, selection_json, parameters_json,
                    frame_json, metadata_json, complete
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

    def get_analysis(
        self,
        analysis_name: str,
        selection: Mapping[str, Any],
        parameters: Mapping[str, Any],
        analysis_version: str = "1",
    ) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        """Return an exact cached analysis result when available."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT frame_json, metadata_json FROM analysis_results
                WHERE analysis_name = ? AND analysis_version = ?
                  AND selection_hash = ? AND parameters_hash = ?
                  AND complete = 1
                """,
                (
                    analysis_name,
                    analysis_version,
                    stable_hash(selection),
                    stable_hash(parameters),
                ),
            ).fetchone()
        if row is None:
            return None
        return pd.read_json(StringIO(row[0]), orient="table"), json.loads(row[1])

    def list_analyses(self) -> pd.DataFrame:
        """List complete cached analyses, including their original specifications."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT analysis_name, analysis_version, selection_hash,
                       parameters_hash, selection_json, parameters_json,
                       metadata_json
                FROM analysis_results WHERE complete = 1
                ORDER BY analysis_name, analysis_version
                """
            ).fetchall()
        return pd.DataFrame(
            rows,
            columns=[
                "analysis_name",
                "analysis_version",
                "selection_hash",
                "parameters_hash",
                "selection_json",
                "parameters_json",
                "metadata_json",
            ],
        )

    def load_analysis(
        self,
        analysis_name: str,
        selection_hash: str,
        parameters_hash: str,
        analysis_version: str = "1",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load one cached result without requiring raw experiment files."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT frame_json, metadata_json FROM analysis_results
                WHERE analysis_name = ? AND analysis_version = ?
                  AND selection_hash = ? AND parameters_hash = ?
                  AND complete = 1
                """,
                (
                    analysis_name,
                    analysis_version,
                    selection_hash,
                    parameters_hash,
                ),
            ).fetchone()
        if row is None:
            raise KeyError("Cached analysis result was not found")
        return pd.read_json(StringIO(row[0]), orient="table"), json.loads(row[1])
