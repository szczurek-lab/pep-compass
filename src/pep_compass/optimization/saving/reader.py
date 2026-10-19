"""Lazy, backend-independent access to one versioned run bundle."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

import pandas as pd

from pep_compass.optimization.saving.backends import (
    FORMAT_VERSION,
    SQLiteStorageBackend,
    canonical_json,
)
from pep_compass.optimization.saving.batches import BatchStore
from pep_compass.utils.logger import get_custom_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import TracebackType
    from typing import Self

    import torch

    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.data_schemas.candidates import CandidateBatch
    from pep_compass.optimization.saving.replay import ReplayResult


logger = get_custom_logger(__name__)


class DataUnavailableError(LookupError):
    """The requested data category was not retained by this run."""


class RunTableHandle:
    """Implement the shared TableHandle protocol with bounded SQLite scans."""

    def __init__(self, backend: SQLiteStorageBackend, name: str) -> None:
        """Initialize this object with the declared configuration.

        :param backend: Backend.
        :type backend: SQLiteStorageBackend
        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :rtype: None
        """
        self.backend, self.name = backend, name

    @property
    def columns(self) -> tuple[str, ...]:
        """Return validated column names from the fixed logical schema.

        :return: Result of this operation.
        :rtype: tuple[str, ...]
        """
        return tuple(
            row[1] for row in self.backend.execute(f'PRAGMA table_info("{self.name}")')
        )

    def count_rows(self) -> int:
        """Count rows without loading payload columns.

        :return: Result of this operation.
        :rtype: int
        """
        return self.backend.execute(f'SELECT COUNT(*) FROM "{self.name}"').fetchone()[0]

    def scan(
        self, columns: Sequence[str], *, chunk_size: int = 100_000
    ) -> Iterator[pd.DataFrame]:
        """Yield requested columns in bounded chunks in storage order.

        :param columns: Column names to project in the requested order.
        :type columns: Sequence[str]
        :param chunk_size: Maximum rows materialized in one scan chunk.
        :type chunk_size: int
        :return: Result of this operation.
        :rtype: Iterator[pd.DataFrame]
        """
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer.")
        if not columns or set(columns) - set(self.columns):
            raise ValueError("Requested columns are not present in this table.")
        selected = ",".join('"' + column + '"' for column in columns)
        cursor = self.backend.execute(f'SELECT {selected} FROM "{self.name}"')
        while rows := cursor.fetchmany(chunk_size):
            yield pd.DataFrame.from_records(
                [tuple(row) for row in rows], columns=columns
            )


class AnalysisReader:
    """Read retained data without reconstructing computations implicitly.

    :param source: Run backend or directory with settings.json/results.sqlite.
    :raises ValueError: If versions, configuration digest or run identities differ.
    """

    def __init__(self, source: SQLiteStorageBackend | str | Path) -> None:
        """Initialize this object with the declared configuration.

        :param source: Source.
        :type source: SQLiteStorageBackend | str | Path
        :rtype: None
        """
        if isinstance(source, SQLiteStorageBackend):
            self.backend = source
        else:
            path = Path(source).resolve()
            settings = json.loads((path / "settings.json").read_text(encoding="utf-8"))
            backend = object.__new__(SQLiteStorageBackend)
            backend.path, backend.settings, backend.lock = path, settings, RLock()
            backend.closed = False
            backend.connection = sqlite3.connect(
                (path / "results.sqlite").as_uri() + "?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            backend.connection.row_factory = sqlite3.Row
            self.backend = backend
        if self.backend.execute("PRAGMA user_version").fetchone()[0] != FORMAT_VERSION:
            raise ValueError("Unsupported run database version.")
        row = self.backend.execute("SELECT * FROM runs").fetchone()
        digest = hashlib.sha256(
            canonical_json(self.backend.settings).encode()
        ).hexdigest()
        if (
            row is None
            or row["settings_sha256"] != digest
            or row["run_id"] != self.backend.settings["run_id"]
        ):
            raise ValueError("Run settings/database identity mismatch.")
        logger.debug("Opened lazy run reader path=%s.", self.backend.path)

    @property
    def settings(self) -> dict[str, object]:
        """Return the resolved run manifest as independent containers.

        :return: Result of this operation.
        :rtype: dict[str, object]
        """
        return json.loads(canonical_json(self.backend.settings))

    @property
    def summary(self) -> dict[str, object]:
        """Return status and aggregate statistics without loading candidates.

        :return: Result of this operation.
        :rtype: dict[str, object]
        """
        row = self.backend.execute("SELECT status,summary_json FROM runs").fetchone()
        result = {"status": row[0], **json.loads(row[1])}
        objective = result.get("objective_name")
        best = self.get_best_candidate(objective) if objective is not None else None
        return {
            **result,
            "best_sequence": best["sequence"] if best else None,
            "best_score": best["score"] if best else None,
        }

    @property
    def tables(self) -> dict[str, RunTableHandle]:
        """Expose the shared logical tables through lazy TableHandle objects.

        :return: Result of this operation.
        :rtype: dict[str, RunTableHandle]
        """
        names = self.backend.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
        ).fetchall()
        return {row[0]: RunTableHandle(self.backend, row[0]) for row in names}

    def get_archive(self, objective: str | None = None) -> pd.DataFrame:
        """Read every evaluated sequence, with the latest score for each objective.

        :param objective: Objective field identity shared by scoring and optimization.
        :type objective: str | None
        :return: Result of this operation.
        :rtype: pd.DataFrame
        """
        return self._evaluations_frame("evaluation_archive", objective)

    def get_evaluations(self, objective: str | None = None) -> pd.DataFrame:
        """Read all actual evaluations; recording is unconditional.

        :param objective: Objective field identity shared by scoring and optimization.
        :type objective: str | None
        :return: Result of this operation.
        :rtype: pd.DataFrame
        """
        return self._evaluations_frame("evaluation_records", objective)

    def _evaluations_frame(self, view: str, objective: str | None) -> pd.DataFrame:
        """Evaluations frame.

        :param view: View.
        :type view: str
        :param objective: Objective field identity shared by scoring and optimization.
        :type objective: str | None
        :return: Result of this operation.
        :rtype: pd.DataFrame
        """
        statement = f"SELECT * FROM {view}"
        parameters = ()
        if objective is not None:
            statement += " WHERE objective=?"
            parameters = (objective,)
        statement += " ORDER BY evaluation_index"
        cursor = self.backend.execute(statement, parameters)
        return pd.DataFrame.from_records(
            [tuple(row) for row in cursor],
            columns=[column[0] for column in cursor.description],
        )

    def _load_batch(self, batch_id: int) -> CandidateBatch:
        """Load batch.

        :param batch_id: Retained ordered batch identity.
        :type batch_id: int
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        try:
            return BatchStore(self.backend, False).load(batch_id)
        except LookupError as error:
            raise DataUnavailableError(
                "Complete batch data was not retained."
            ) from error

    def _load_checkpoint(self, payload_id: str) -> dict[str, object]:
        """Restore a checkpoint using its historical evaluation-table prefix.

        :param payload_id: Content hash of a supported checkpoint payload.
        :type payload_id: str
        :return: State and batch ready for explicit fragment reconstruction.
        :rtype: dict[str, object]
        :raises ValueError: If the checkpoint payload is not a mapping.
        """
        checkpoint = self.backend.load_payload(payload_id)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint mapping.")
        observations = {}
        directions = {}
        for objective, sequence, score, direction in self.backend.execute(
            "SELECT objective,sequence,score,direction FROM evaluation_records WHERE evaluation_index<? ORDER BY evaluation_index",
            (checkpoint["evaluation_count"],),
        ):
            observations.setdefault(objective, {})[sequence] = score
            directions[objective] = direction
        checkpoint["state"]["observations"] = observations
        checkpoint["state"]["objective_directions"] = directions
        checkpoint["batch"] = self._load_batch(checkpoint["batch_id"])
        return checkpoint

    def get_final_batch(self) -> CandidateBatch:
        """Read the actual root output retained by reconstruction='trajectories'.

        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        if (
            self.settings["runner_settings"]["storage"]["reconstruction"]
            != "trajectories"
        ):
            raise DataUnavailableError(
                "Enable reconstruction='trajectories' to retain the graph output."
            )
        row = self.backend.execute(
            "SELECT output_batch_id FROM step_executions WHERE parent_execution_id IS NULL AND phase='execute' AND status='completed' ORDER BY execution_id DESC LIMIT 1"
        ).fetchone()
        if row is None or row[0] is None:
            raise DataUnavailableError("No completed graph output was retained.")
        return self._load_batch(row[0])

    def get_snapshot(self, name: str) -> list[CandidateBatch]:
        """Read labeled references through the same ordered batch representation.

        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :return: Result of this operation.
        :rtype: list[CandidateBatch]
        """
        rows = self.backend.execute(
            "SELECT batch_id FROM candidate_sets WHERE name=? ORDER BY set_id", (name,)
        ).fetchall()
        if not rows:
            raise DataUnavailableError(f"No retained snapshot named {name!r}.")
        return [self._load_batch(row[0]) for row in rows]

    def get_lineage(self) -> pd.DataFrame:
        """Read structural provenance, including unscored ancestor identities.

        :return: Result of this operation.
        :rtype: pd.DataFrame
        """
        if not self.settings["runner_settings"]["storage"]["save_lineage"]:
            raise DataUnavailableError("Lineage was not retained.")
        return self._table_frame("batch_lineage")

    def get_metadata(self) -> dict[str, object]:
        """Return configuration and run status metadata.

        :return: Result of this operation.
        :rtype: dict[str, object]
        """
        return {"settings": self.settings, "summary": self.summary}

    def get_best_candidate(
        self, objective: str | None = None
    ) -> dict[str, object] | None:
        """Return the best retained evaluation for one explicitly selected objective.

        :param objective: Objective field identity shared by scoring and optimization.
        :type objective: str | None
        :return: Result of this operation.
        :rtype: dict[str, object] | None
        """
        if objective is None:
            names = self.backend.execute("SELECT objective FROM objectives").fetchall()
            if len(names) > 1:
                raise ValueError(
                    "Select an objective when multiple objectives were evaluated."
                )
            if not names:
                return None
            objective = names[0][0]
        direction = self.backend.execute(
            "SELECT direction FROM objectives WHERE objective=?", (objective,)
        ).fetchone()
        if direction is None:
            return None
        ordering = "DESC" if direction[0] == "maximize" else "ASC"
        row = self.backend.execute(
            f"SELECT * FROM evaluation_archive WHERE objective=? ORDER BY score {ordering},evaluation_index LIMIT 1",
            (objective,),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_component_data(self) -> pd.DataFrame:
        """Read optional component diagnostics on explicit request.

        :return: Result of this operation.
        :rtype: pd.DataFrame
        """
        if self.settings["runner_settings"]["storage"]["component_data"] == "none":
            raise DataUnavailableError("Component diagnostics were not retained.")
        result = self._table_frame("component_records")

        def load_record(payload_id: str) -> object:
            """Load record.

            :param payload_id: Content hash identifying one retained typed payload.
            :type payload_id: str
            :return: Result of this operation.
            :rtype: object
            """
            value = self.backend.load_payload(payload_id)
            if isinstance(value, dict) and "field_payloads" in value:
                return {
                    "display": value["display"],
                    "fields": {
                        name: self.backend.load_payload(identity)
                        for name, identity in value["field_payloads"].items()
                    },
                }
            return value

        result["data"] = result["payload_id"].map(load_record)
        return result

    def _table_frame(self, name: str) -> pd.DataFrame:
        """Table frame.

        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :return: Result of this operation.
        :rtype: pd.DataFrame
        """
        table = self.tables[name]
        chunks = list(table.scan(table.columns))
        return (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else pd.DataFrame(columns=table.columns)
        )

    def get_checkpoints(self) -> pd.DataFrame:
        """Return retained replay boundaries without loading their tensor payloads.

        :return: Result of this operation.
        :rtype: pd.DataFrame
        """
        return self._table_frame("checkpoints")

    def get_boundary_input(self, execution_id: int) -> CandidateBatch:
        """Load the input CandidateBatch retained for one operation invocation.

        :param execution_id: Step execution identity from get_checkpoints().
        :return: Original aligned input batch with latent origins and fields.
        :raises DataUnavailableError: If no reconstruction boundary was retained.

        :type execution_id: int
        :rtype: CandidateBatch
        """
        row = self.backend.execute(
            "SELECT payload_id FROM checkpoints WHERE execution_id=?",
            (int(execution_id),),
        ).fetchone()
        if row is None:
            raise DataUnavailableError("No retained boundary for this execution.")
        checkpoint = self.backend.load_payload(row[0])
        return self._load_batch(checkpoint["batch_id"])

    def get_trajectories(self) -> tuple[pd.DataFrame, torch.Tensor]:
        """Load walker-produced sequences, metadata and aligned latent positions.

        :return: Pair of a metadata DataFrame and a tensor shaped (N, D).
        :raises DataUnavailableError: If trajectory retention was disabled.

        :rtype: tuple[pd.DataFrame, torch.Tensor]
        """
        import torch

        from pep_compass.data_schemas.candidates import (
            ObjectField,
            OptionalField,
            TensorField,
        )

        if (
            self.settings["runner_settings"]["storage"]["reconstruction"]
            != "trajectories"
        ):
            raise DataUnavailableError(
                "Enable reconstruction='trajectories' to retain walk points."
            )
        rows, latents = [], []
        cursor = self.backend.execute(
            "SELECT execution_id,batch_id FROM candidate_sets WHERE kind='trajectory' ORDER BY set_id"
        )
        for execution_id, batch_id in cursor:
            batch = self._load_batch(batch_id)
            for index, sequence in enumerate(batch.sequences):
                record = {
                    "execution_id": execution_id,
                    "sequence": sequence,
                    "latent_index": len(rows),
                }
                for name in (
                    "tracking.trajectory_id",
                    "tracking.trajectory_index",
                    "tracking.trajectory_step",
                    "lineage.candidate_id",
                ):
                    field = batch.fields.get(name)
                    if isinstance(field, OptionalField):
                        field = field.values if bool(field.valid[index]) else None
                    if isinstance(field, (TensorField, ObjectField)):
                        value = field.values[index]
                        record[name.split(".")[-1]] = (
                            value.item() if isinstance(value, torch.Tensor) else value
                        )
                rows.append(record)
            latents.append(batch.latent_origins)
        frame = pd.DataFrame(
            rows,
            columns=[
                "execution_id",
                "sequence",
                "latent_index",
                "trajectory_id",
                "trajectory_index",
                "trajectory_step",
                "candidate_id",
            ],
        )
        return frame, torch.cat(latents, dim=0) if latents else torch.empty(
            (0, 0)
        )  # (N, D)

    def reconstruct_local_enumeration(
        self, checkpoint_id: int, *, autoencoder: Autoencoder | None = None
    ) -> ReplayResult:
        """Re-execute one retained boundary and verify ordered output sequences.

        :param checkpoint_id: Retained replay checkpoint identity.
        :type checkpoint_id: int
        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder | None
        :return: Result of this operation.
        :rtype: ReplayResult
        """
        from pep_compass.optimization.saving.replay import reconstruct

        return reconstruct(self, checkpoint_id, autoencoder=autoencoder)

    def close(self) -> None:
        """Release the reader-owned connection, including a memory database.

        :rtype: None
        """
        self.backend.close()

    def __enter__(self) -> Self:
        """Return this lazy reader for deterministic resource cleanup.

        :return: Result of this operation.
        :rtype: Self
        """
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Finalize the managed scope and propagate exceptions.

        :param kind: Kind of recorded artifact or exception type, as declared by this operation.
        :type kind: type[BaseException] | None
        :param value: Value consumed by this conversion or transformation.
        :type value: BaseException | None
        :param traceback: Original exception traceback, if present.
        :type traceback: TracebackType | None
        :return: Result of this operation.
        :rtype: bool
        """
        self.close()
        return False
