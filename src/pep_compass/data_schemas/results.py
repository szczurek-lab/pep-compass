"""Shared computation and run-result contracts with lazy storage references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from pep_compass.optimization.saving.reader import AnalysisReader


@dataclass(frozen=True)
class RunResult:
    """One completed run and a lazy view over exactly its retained information."""

    summary: dict
    reader: AnalysisReader
    path: Path | None

    @property
    def archive(self) -> pd.DataFrame:
        """Return the base archive of evaluated peptides.

        :return: Result of this operation.
        :rtype: pd.DataFrame
        """
        return self.reader.get_archive()

    def save(self, path: str | Path) -> Path:
        """Export only retained information as settings.json and results.sqlite.

        :param path: Input or output path used by this operation.
        :type path: str | Path
        :return: Result of this operation.
        :rtype: Path
        """
        return self.reader.backend.export(path)

    def close(self) -> None:
        """Release lazy result resources after all reads are complete.

        :rtype: None
        """
        self.reader.close()
