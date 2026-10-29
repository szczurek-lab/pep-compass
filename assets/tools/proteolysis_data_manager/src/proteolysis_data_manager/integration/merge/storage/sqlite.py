"""Transactional SQLite serialization and a single public dataset reader."""

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from proteolysis_data_manager.integration.merge.contract.dataset import UnifiedDataset
from proteolysis_data_manager.integration.merge.contract.schema import (
    FOREIGN_KEYS,
    PRIMARY_KEYS,
    QUANTITIES,
    SCHEMA_VERSION,
    TABLES,
    canonical,
)
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def write_dataset(dataset, path, *, replace=False):
    """Validate and atomically write one versioned SQLite database.

    :param dataset: Complete common dataset.
    :param path: SQLite destination, outside acquired raw directories.
    :param bool replace: Explicit permission to replace an existing derived database.
    :return: Destination path.
    :raises ValueError: If validation fails or the destination is inside raw.
    :raises FileExistsError: If replacement was not explicitly selected.
    """
    path = Path(path)
    if "raw" in path.resolve().parts:
        raise ValueError("The unified database must be outside raw")
    if path.exists() and not replace:
        raise FileExistsError(path)
    dataset.validate()
    logger.info("Writing common dataset to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".unified-", suffix=".sqlite", dir=path.parent
    )
    os.close(descriptor)
    try:
        with sqlite3.connect(temporary) as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)"
            )
            db.executemany(
                "INSERT INTO metadata VALUES (?,?)",
                [
                    ("schema_version", SCHEMA_VERSION),
                    ("build", canonical(dataset.metadata)),
                ],
            )
            db.execute(
                "CREATE TABLE quantities (quantity TEXT PRIMARY KEY,unit TEXT,signed INTEGER,definition TEXT NOT NULL)"
            )
            db.executemany(
                "INSERT INTO quantities VALUES (?,?,?,?)",
                [
                    (name, unit, int(signed), definition)
                    for name, (unit, signed, definition) in QUANTITIES.items()
                ],
            )
            for name, columns in TABLES.items():
                definitions = [f'"{column}" {kind}' for column, kind in columns.items()]
                definitions.append("PRIMARY KEY (" + ",".join(PRIMARY_KEYS[name]) + ")")
                definitions.extend(
                    f"FOREIGN KEY ({column}) REFERENCES {target}({key})"
                    for column, (target, key) in FOREIGN_KEYS.get(name, {}).items()
                )
                db.execute(f'CREATE TABLE "{name}" (' + ",".join(definitions) + ")")
                dataset.table(name).to_sql(
                    name, db, if_exists="append", index=False, chunksize=10000
                )
            for table, column in [
                ("source_records", "source_db"),
                ("source_records", "source_dataset"),
                ("observations", "quantity"),
                ("observations", "source_record_id"),
                ("contexts", "source_record_id"),
                ("participants", "assay_id"),
                ("profile_cells", "profile_id"),
            ]:
                db.execute(
                    f'CREATE INDEX "idx_{table}_{column}" ON "{table}" ("{column}")'
                )
            if db.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("SQLite foreign-key validation failed")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


class ProteolysisDataset(UnifiedDataset):
    """Read the same common tables regardless of which parsers produced them."""

    @classmethod
    def read(cls, path):
        """Open an existing database read-only and validate its version and tables.

        :param path: Unified SQLite database path.
        :return: Validated common dataset with shared public views.
        :raises ValueError: If schema, registry or foreign keys differ.
        """
        path = Path(path).resolve()
        logger.info("Reading common dataset from %s", path)
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            meta = dict(db.execute("SELECT key,value FROM metadata"))
            if meta.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Unsupported unified schema version")
            quantities = {
                row[0]: row[1:]
                for row in db.execute(
                    "SELECT quantity,unit,signed,definition FROM quantities"
                )
            }
            expected = {
                name: (unit, int(signed), definition)
                for name, (unit, signed, definition) in QUANTITIES.items()
            }
            if quantities != expected:
                raise ValueError("Incorrect quantity registry")
            if db.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Invalid database foreign keys")
            tables = {
                name: pd.read_sql_query(f'SELECT * FROM "{name}"', db)
                for name in TABLES
            }
        return cls(tables, json.loads(meta["build"])).validate()
