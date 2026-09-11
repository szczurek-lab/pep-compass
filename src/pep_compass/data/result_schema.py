"""Versioned schema of persisted PepCompass experiment results."""

from __future__ import annotations

from dataclasses import dataclass


CURRENT_RESULT_SCHEMA_VERSION = "2"


@dataclass(frozen=True, slots=True)
class ResultFileSchema:
    """Describe one table written for an experiment result.

    :param name: Stable logical table name.
    :param relative_path: Path relative to one run directory.
    :param required_columns: Columns required from this schema version.
    """

    name: str
    relative_path: str
    required_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResultSchema:
    """Describe the physical files produced by one schema version."""

    version: str
    tables: tuple[ResultFileSchema, ...]

    def table(self, name: str) -> ResultFileSchema:
        """Return a named table definition.

        :param name: Logical table name.
        :return: Matching table definition.
        :raises KeyError: If the schema does not define the table.
        """
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(name)


RESULT_SCHEMA = ResultSchema(
    version=CURRENT_RESULT_SCHEMA_VERSION,
    tables=(
        ResultFileSchema(
            "steps",
            "tracking/steps.csv",
            (
                "execution_id",
                "step_name",
                "path",
                "input_size",
                "output_size",
                "duration_seconds",
            ),
        ),
        ResultFileSchema(
            "trajectory_points",
            "tracking/trajectory_points.csv",
            (
                "execution_id",
                "trajectory_id",
                "trajectory_step",
                "latent_index",
            ),
        ),
        ResultFileSchema(
            "local_enumerations",
            "tracking/local_enumerations.csv",
            (
                "execution_id",
                "input_count",
                "output_count",
                "output_sequences_sha256",
            ),
        ),
        ResultFileSchema(
            "candidates",
            "tracking/candidates.csv",
            ("execution_id", "candidate_index", "sequence"),
        ),
    ),
)
