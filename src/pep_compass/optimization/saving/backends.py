"""Shared relational storage in RAM or a persistent SQLite run bundle."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from pep_compass.optimization.saving.schema import FORMAT_VERSION, SCHEMA
from pep_compass.optimization.saving.serialization import dumps, loads
from pep_compass.utils.logger import get_custom_logger

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = get_custom_logger(__name__)


def canonical_json(value: object) -> str:
    """Serialize run settings deterministically, rejecting opaque objects.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: str
    """
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


class SQLiteStorageBackend:
    """Persist a single run with transaction batching and bounded table access.

    :param settings: Fully resolved, JSON-compatible run manifest.
    :param path: New output directory, or ``None`` for a memory database.
    :raises FileExistsError: If output would overwrite an existing run bundle.
    """

    def __init__(
        self, settings: dict[str, object], path: str | Path | None = None
    ) -> None:
        """Initialize this object with the declared configuration.

        :param settings: Validated settings or a resolved settings manifest.
        :type settings: dict[str, object]
        :param path: Input or output path used by this operation.
        :type path: str | Path | None
        :rtype: None
        """
        self.path = Path(path).resolve() if path is not None else None
        self.lock = RLock()
        self.closed = False
        self.settings = settings
        serialized = canonical_json(settings)
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
            if (self.path / "results.sqlite").exists() or (
                self.path / "settings.json"
            ).exists():
                raise FileExistsError(f"Run bundle already exists: {self.path}")
        self.connection = sqlite3.connect(
            str(self.path / "results.sqlite") if self.path is not None else ":memory:",
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        self.connection.execute(f"PRAGMA user_version = {FORMAT_VERSION}")
        self.connection.execute(
            "INSERT INTO runs VALUES (?,?,?,?)",
            (
                settings["run_id"],
                "running",
                hashlib.sha256(serialized.encode()).hexdigest(),
                "{}",
            ),
        )
        if self.path is not None:
            temporary = self.path / "settings.json.tmp"
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(self.path / "settings.json")
        self.connection.commit()
        logger.info(
            "Opened run storage backend=%s path=%s.",
            "sqlite" if self.path is not None else "memory",
            self.path,
        )

    def execute(
        self, statement: str, parameters: Sequence[object] = ()
    ) -> sqlite3.Cursor:
        """Execute an internal parameterized statement under the backend lock.

        :param statement: Statement.
        :type statement: str
        :param parameters: Named configuration values consumed by this operation.
        :type parameters: Sequence[object]
        :return: Result of this operation.
        :rtype: sqlite3.Cursor
        """
        with self.lock:
            return self.connection.execute(statement, parameters)

    def payload(self, value: object) -> str:
        """Store a content-addressed, typed payload and return its identity.

        :param value: Value consumed by this conversion or transformation.
        :type value: object
        :return: Result of this operation.
        :rtype: str
        """
        data = dumps(value)
        identity = hashlib.sha256(data).hexdigest()
        self.execute(
            "INSERT OR IGNORE INTO tensor_payloads VALUES (?,?,?)", (identity, 1, data)
        )
        return identity

    def load_payload(self, identity: str) -> object:
        """Read one explicitly requested payload and check its checksum.

        :param identity: Identity.
        :type identity: str
        :return: Result of this operation.
        :rtype: object
        """
        row = self.execute(
            "SELECT data FROM tensor_payloads WHERE payload_id=?", (identity,)
        ).fetchone()
        if row is None:
            raise KeyError(identity)
        data = bytes(row[0])
        if hashlib.sha256(data).hexdigest() != identity:
            raise ValueError("Run payload checksum mismatch.")
        return loads(data)

    def flush(self) -> None:
        """Commit every buffered record without dropping intermediate events.

        :rtype: None
        """
        with self.lock:
            self.connection.commit()

    def export(self, path: str | Path) -> Path:
        """Copy retained data and immutable settings to a new run bundle.

        :param path: Input or output path used by this operation.
        :type path: str | Path
        :return: Result of this operation.
        :rtype: Path
        """
        target = Path(path).resolve()
        target.mkdir(parents=True, exist_ok=True)
        if (target / "results.sqlite").exists() or (target / "settings.json").exists():
            raise FileExistsError(target)
        self.flush()
        with self.lock, sqlite3.connect(target / "results.sqlite") as destination:
            self.connection.backup(destination)
        (target / "settings.json").write_text(
            canonical_json(self.settings), encoding="utf-8"
        )
        logger.info("Exported retained run data path=%s.", target)
        return target

    def close(self) -> None:
        """Commit pending records and release the storage connection.

        :rtype: None
        """
        with self.lock:
            if self.closed:
                return
            self.connection.commit()
            self.connection.close()
            self.closed = True
