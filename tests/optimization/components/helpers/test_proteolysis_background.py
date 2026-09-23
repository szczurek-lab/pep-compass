"""Tests for the residue background distributions."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from pep_compass.optimization.components.helpers.proteolysis import MEROPS_AMINO_ACIDS
from pep_compass.optimization.components.helpers.proteolysis.background import (
    amino_acid_background_from_fasta,
    load_amino_acid_background,
    normalize_background,
    uniform_background,
)


N_AMINO_ACIDS = len(MEROPS_AMINO_ACIDS)


def _write_fasta(path: Path, records: dict[str, str], compress: bool = False) -> Path:
    """Write a small FASTA file.

    :param path: Destination path.
    :param records: Mapping from identifier to sequence.
    :param compress: Whether to gzip the output.
    :return: Path of the written file.
    """

    text = "".join(f">{name}\n{sequence}\n" for name, sequence in records.items())
    if compress:
        path = path.with_suffix(path.suffix + ".gz")
        path.write_bytes(gzip.compress(text.encode("utf-8")))
    else:
        path.write_text(text, encoding="utf-8")
    return path


class TestNormalizeBackground:
    """Coercion of a background argument into a distribution."""

    def test_none_resolves_to_uniform(self) -> None:
        """A missing background is the uniform distribution."""

        assert np.allclose(normalize_background(None, N_AMINO_ACIDS), uniform_background())

    def test_unnormalized_input_is_rescaled(self) -> None:
        """Counts may be passed instead of frequencies."""

        counts = np.arange(1, N_AMINO_ACIDS + 1, dtype=float)
        result = normalize_background(counts, N_AMINO_ACIDS)
        assert result.sum() == pytest.approx(1.0)
        assert np.allclose(result, counts / counts.sum())

    def test_wrong_length_is_rejected(self) -> None:
        """A background of the wrong alphabet size is a programming error."""

        with pytest.raises(ValueError):
            normalize_background(np.ones(5), N_AMINO_ACIDS)

    def test_zero_entry_is_rejected(self) -> None:
        """A zero reference frequency makes the log-ratio undefined."""

        values = np.ones(N_AMINO_ACIDS)
        values[3] = 0.0
        with pytest.raises(ValueError):
            normalize_background(values, N_AMINO_ACIDS)


class TestBackgroundFromFasta:
    """Residue composition computed from a FASTA file."""

    def test_composition_matches_the_residue_counts(self, tmp_path: Path) -> None:
        """Frequencies follow the counted residues in alphabet order."""

        path = _write_fasta(tmp_path / "toy.fasta", {"a": "AAAC", "b": "AC"})
        frequencies, counts = amino_acid_background_from_fasta(path)
        assert counts["n_sequences"] == 2
        assert counts["A"] == 4
        assert counts["C"] == 2
        assert frequencies[MEROPS_AMINO_ACIDS.index("A")] == pytest.approx(4 / 6)
        assert frequencies[MEROPS_AMINO_ACIDS.index("C")] == pytest.approx(2 / 6)
        assert frequencies.sum() == pytest.approx(1.0)

    def test_residues_outside_the_alphabet_are_counted_separately(self, tmp_path: Path) -> None:
        """Non-standard residues are excluded from the distribution but reported.

        The MEROPS matrices have no column for them, so including them would
        change the alphabet the metrics operate on.
        """

        path = _write_fasta(tmp_path / "toy.fasta", {"a": "AAXU"})
        frequencies, counts = amino_acid_background_from_fasta(path)
        assert counts["non_standard"] == 2
        assert frequencies[MEROPS_AMINO_ACIDS.index("A")] == pytest.approx(1.0)

    def test_gzip_input_is_read(self, tmp_path: Path) -> None:
        """A compressed snapshot does not need to be expanded first."""

        path = _write_fasta(tmp_path / "toy.fasta", {"a": "ACDE"}, compress=True)
        frequencies, counts = amino_acid_background_from_fasta(path)
        assert counts["n_sequences"] == 1
        assert frequencies[MEROPS_AMINO_ACIDS.index("D")] == pytest.approx(0.25)

    def test_file_without_standard_residues_is_rejected(self, tmp_path: Path) -> None:
        """An empty composition cannot serve as a reference distribution."""

        path = _write_fasta(tmp_path / "toy.fasta", {"a": "XXXX"})
        with pytest.raises(ValueError):
            amino_acid_background_from_fasta(path)


class TestLoadAminoAcidBackground:
    """Reading the stored composition of the acquisition script."""

    def test_round_trip_through_the_stored_format(self, tmp_path: Path) -> None:
        """The loader reproduces the frequencies in alphabet order."""

        frequencies = np.linspace(1.0, 2.0, N_AMINO_ACIDS)
        frequencies = frequencies / frequencies.sum()
        path = tmp_path / "composition.json"
        path.write_text(
            json.dumps(
                {"frequencies": dict(zip(MEROPS_AMINO_ACIDS, frequencies.tolist()))},
                indent=2,
            ),
            encoding="utf-8",
        )
        assert np.allclose(load_amino_acid_background(path), frequencies)

    def test_missing_residue_is_reported(self, tmp_path: Path) -> None:
        """A composition lacking a residue cannot be silently completed."""

        path = tmp_path / "composition.json"
        path.write_text(json.dumps({"frequencies": {"A": 1.0}}), encoding="utf-8")
        with pytest.raises(KeyError):
            load_amino_acid_background(path)
