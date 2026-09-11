"""Materialize independent input tasks from inline or CSV sources."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class InputTask:
    """Represent one independently executable input repetition."""

    index: int
    sequence: str
    source_index: int
    repetition: int

    @property
    def task_id(self) -> str:
        """Return a stable filesystem-safe task identifier."""
        return f"task_{self.index:05d}"


def materialize_input_tasks(
    configuration: Mapping[str, Any],
    *,
    working_directory: Path,
) -> tuple[InputTask, ...]:
    """Materialize deterministic tasks from inline or CSV input.

    :param configuration: Experiment input mapping.
    :param working_directory: Base for relative CSV paths.
    :return: Input repetitions in source order.
    """
    inline = configuration.get("sequences")
    csv_configuration = configuration.get("csv")
    if (inline is None) == (csv_configuration is None):
        raise ValueError("Configure exactly one of input.sequences or input.csv.")
    if inline is not None:
        if not isinstance(inline, Sequence) or isinstance(inline, (str, bytes)):
            raise ValueError("input.sequences must be a sequence.")
        repetitions = _positive_integer(configuration.get("repetitions", 1))
        sources = [(str(sequence), repetitions) for sequence in inline]
    else:
        if not isinstance(csv_configuration, Mapping):
            raise ValueError("input.csv must be a mapping.")
        sources = _load_csv_sources(csv_configuration, working_directory)
    if not sources or any(not sequence for sequence, _ in sources):
        raise ValueError("Input must contain non-empty sequences.")
    tasks = []
    for source_index, (sequence, repetitions) in enumerate(sources):
        for repetition in range(repetitions):
            tasks.append(
                InputTask(len(tasks), sequence, source_index, repetition)
            )
    return tuple(tasks)


def _load_csv_sources(
    configuration: Mapping[str, Any],
    working_directory: Path,
) -> list[tuple[str, int]]:
    """Read sequence and repetition pairs from one CSV file."""
    path = Path(str(configuration.get("path", "")))
    if not path.name:
        raise ValueError("input.csv.path cannot be empty.")
    if not path.is_absolute():
        path = working_directory / path
    sequence_column = str(configuration.get("sequence_column", "sequence"))
    repetitions_column = str(
        configuration.get("repetitions_column", "repetitions")
    )
    sources = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or sequence_column not in reader.fieldnames:
            raise ValueError(f"Input CSV requires column: {sequence_column}")
        for row_number, row in enumerate(reader, start=2):
            sequence = (row.get(sequence_column) or "").strip()
            if not sequence:
                raise ValueError(f"Input CSV row {row_number} has an empty sequence.")
            repetitions = _positive_integer(row.get(repetitions_column) or 1)
            sources.append((sequence, repetitions))
    return sources


def _positive_integer(value: Any) -> int:
    """Parse a strictly positive integer."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Repetitions must be a positive integer.") from error
    if parsed < 1:
        raise ValueError("Repetitions must be a positive integer.")
    return parsed
