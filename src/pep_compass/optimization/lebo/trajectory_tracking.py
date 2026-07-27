"""CSV tracking for LE-BO candidate provenance and iteration summaries."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TrackingLevel = Literal["short", "normal", "full"]

EVALUATION_FIELDS = [
    "run_id",
    "iteration_id",
    "candidate_id",
    "sequence",
    "objective_name",
    "objective_value",
    "objective_direction",
]


@dataclass
class CandidateProvenance:
    """Describe the first local-enumeration path that produced a candidate.

    :param sequence: Candidate peptide sequence.
    :param parent_sequence: Sequence mutated to create the candidate.
    :param trajectory_id: Zero-based SORBES trajectory index.
    :param step_id: One-based step within the trajectory.
    :param path: Decoded SORBES path up to the generating step.
    """

    sequence: str
    parent_sequence: str
    trajectory_id: int | None
    step_id: int | None
    path: list[str] = field(default_factory=list)


@dataclass
class EnumerationTrace:
    """Summarize one local-enumeration call without retaining rejected peptides."""

    generated_count: int = 0
    accepted_count: int = 0
    candidates: dict[str, CandidateProvenance] = field(default_factory=dict)


class LeboCSVTracker:
    """Persist compact LE-BO histories in separate analysis-friendly files.

    ``short`` stores iteration summaries and evaluated peptides. ``normal`` also
    stores the generating path of evaluated candidates. ``full`` stores every
    candidate that passed local filtering and could enter optimizer selection.
    Rejected sequences are represented only by aggregate counts.
    """

    def __init__(
        self,
        output_directory: str | Path,
        run_id: str,
        level: TrackingLevel,
        candidate_strategy: str,
        objective_name: str,
        objective_direction: str,
        objective_description: str,
        objective_parameters: dict[str, Any],
        encoder_decoder: Any | None = None,
        store_latents: bool = True,
        component_provider: (
            Callable[[list[str]], dict[str, list[float]]] | None
        ) = None,
        component_fields: Sequence[str] = (),
    ) -> None:
        if level not in {"short", "normal", "full"}:
            raise ValueError("tracking.level must be short, normal, or full")
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.level = level
        self.run_id = run_id
        self.candidate_strategy = candidate_strategy
        self.objective_name = objective_name
        self.objective_direction = objective_direction
        self.objective_description = objective_description
        self.objective_parameters = objective_parameters
        self.encoder_decoder = encoder_decoder
        self.store_latents = store_latents
        self.component_provider = component_provider
        self.component_fields = (
            list(component_fields) if component_provider is not None else []
        )
        self._seen_candidates: set[str] = set()
        self._candidate_ids: dict[str, str] = {}
        self._write_headers()
        self._write_metadata()

    def _write_headers(self) -> None:
        self._write_rows(
            "iteration_statistics.csv",
            [
                "run_id",
                "iteration_id",
                "center_sequence",
                "generated_count",
                "accepted_count",
                "rejected_count",
                "evaluated_count",
                "candidate_pool_size",
                "best_sequence",
                "best_objective_value",
            ],
            [],
        )
        self._write_rows("evaluations.csv", self._evaluation_fieldnames(), [])
        if self.level != "short":
            self._write_rows(
                "candidates.csv",
                [
                    "run_id",
                    "iteration_id",
                    "candidate_id",
                    "sequence",
                    "parent_sequence",
                    "trajectory_id",
                    "step_id",
                    "trajectory_path",
                    "latent_point",
                    "evaluated",
                ],
                [],
            )

    def _evaluation_fieldnames(self) -> list[str]:
        """Return the evaluation columns, including objective-specific terms."""
        return [*EVALUATION_FIELDS, *self.component_fields]

    def _write_metadata(self) -> None:
        metadata = {
            "tracking_level": self.level,
            "run_id": self.run_id,
            "candidate_strategy": self.candidate_strategy,
            "objective_name": self.objective_name,
            "objective_direction": self.objective_direction,
            "objective_description": self.objective_description,
            "objective_parameters": self.objective_parameters,
            "store_latents": self.store_latents,
        }
        with (self.output_directory / "tracking_metadata.json").open(
            "w", encoding="utf-8"
        ) as output:
            json.dump(metadata, output, indent=2)

    def _write_rows(
        self, filename: str, fieldnames: list[str], rows: list[dict[str, Any]]
    ) -> None:
        path = self.output_directory / filename
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _encode(self, sequences: list[str]) -> dict[str, str]:
        if not sequences or not self.store_latents or self.encoder_decoder is None:
            return {}
        import torch

        encoded: dict[str, str] = {}
        with torch.no_grad():
            for start in range(0, len(sequences), 256):
                batch = sequences[start : start + 256]
                latent_points = self.encoder_decoder.encode_peptides(batch)
                encoded.update(
                    {
                        sequence: json.dumps(point.detach().cpu().tolist())
                        for sequence, point in zip(batch, latent_points)
                    }
                )
        return encoded

    def _candidate_id(self, sequence: str) -> str:
        """Return one stable run-local identifier for a peptide sequence."""
        if sequence not in self._candidate_ids:
            self._candidate_ids[sequence] = (
                f"candidate_{len(self._candidate_ids):06d}"
            )
        return self._candidate_ids[sequence]

    def _evaluation_rows(
        self, iteration_id: int, evaluations: list[tuple[str, float]]
    ) -> list[dict[str, Any]]:
        """Build evaluation rows and append objective-specific score terms."""
        rows = [
            {
                "run_id": self.run_id,
                "iteration_id": iteration_id,
                "candidate_id": self._candidate_id(sequence),
                "sequence": sequence,
                "objective_name": self.objective_name,
                "objective_value": score,
                "objective_direction": self.objective_direction,
            }
            for sequence, score in evaluations
        ]
        if not rows or not self.component_fields:
            return rows
        components = self.component_provider([sequence for sequence, _ in evaluations])
        for index, row in enumerate(rows):
            row.update(
                {field: components[field][index] for field in self.component_fields}
            )
        return rows

    def record_iteration(
        self,
        iteration_id: int,
        center_sequence: str,
        trace: EnumerationTrace,
        evaluations: list[tuple[str, float]],
        candidate_pool_size: int,
        best_sequence: str,
        best_objective_value: float,
        candidate_provenance: dict[str, CandidateProvenance] | None = None,
    ) -> None:
        """Append one optimizer iteration and its configured candidate subset."""
        evaluated = {sequence for sequence, _ in evaluations}
        self._write_rows(
            "iteration_statistics.csv",
            [
                "run_id",
                "iteration_id",
                "center_sequence",
                "generated_count",
                "accepted_count",
                "rejected_count",
                "evaluated_count",
                "candidate_pool_size",
                "best_sequence",
                "best_objective_value",
            ],
            [
                {
                    "run_id": self.run_id,
                    "iteration_id": iteration_id,
                    "center_sequence": center_sequence,
                    "generated_count": trace.generated_count,
                    "accepted_count": trace.accepted_count,
                    "rejected_count": max(
                        trace.generated_count - trace.accepted_count, 0
                    ),
                    "evaluated_count": len(evaluations),
                    "candidate_pool_size": candidate_pool_size,
                    "best_sequence": best_sequence,
                    "best_objective_value": best_objective_value,
                }
            ],
        )
        self._write_rows(
            "evaluations.csv",
            self._evaluation_fieldnames(),
            self._evaluation_rows(iteration_id, evaluations),
        )
        if self.level == "short":
            return
        provenance = (
            trace.candidates
            if candidate_provenance is None
            else candidate_provenance
        )
        selected = (
            provenance
            if self.level == "full"
            else {
                sequence: provenance[sequence]
                for sequence in evaluated
                if sequence in provenance
            }
        )
        selected = {
            sequence: provenance
            for sequence, provenance in selected.items()
            if sequence not in self._seen_candidates
        }
        latent_points = self._encode(list(selected))
        self._write_rows(
            "candidates.csv",
            [
                "run_id",
                "iteration_id",
                "candidate_id",
                "sequence",
                "parent_sequence",
                "trajectory_id",
                "step_id",
                "trajectory_path",
                "latent_point",
                "evaluated",
            ],
            [
                {
                    "run_id": self.run_id,
                    "iteration_id": iteration_id,
                    "candidate_id": self._candidate_id(sequence),
                    "sequence": provenance.sequence,
                    "parent_sequence": provenance.parent_sequence,
                    "trajectory_id": provenance.trajectory_id,
                    "step_id": provenance.step_id,
                    "trajectory_path": json.dumps(provenance.path),
                    "latent_point": latent_points.get(sequence, ""),
                    "evaluated": sequence in evaluated,
                }
                for sequence, provenance in selected.items()
            ],
        )
        self._seen_candidates.update(selected)
