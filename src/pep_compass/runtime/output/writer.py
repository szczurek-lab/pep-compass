"""Atomic output of run results and stability measurements."""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from pep_compass.data.result_schema import CURRENT_RESULT_SCHEMA_VERSION
from pep_compass.optimization.engine.execution.result import OptimizationResult
from pep_compass.optimization.stability_estimation.monitoring import MemorySnapshot
from pep_compass.runtime.planning.plan import PlannedRun
from pep_compass.runtime.configuration.schema import RuntimeConfiguration


class ResultWriter:
    """Write independently recoverable run results below one output root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def run_directory(self, entry: PlannedRun) -> Path:
        """Return the stable output directory for a plan entry."""
        return self.root / "variants" / entry.variant.variant_id / "runs" / entry.run_id

    def is_completed(self, entry: PlannedRun) -> bool:
        """Return whether a plan entry has a durable completed status."""
        status_path = self.run_directory(entry) / "result.json"
        if not status_path.exists():
            return False
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))["status"] == "completed"
        except (KeyError, json.JSONDecodeError):
            return False

    def write_running(self, entry: PlannedRun) -> None:
        """Persist a running status before computation starts."""
        self._write_status(entry, "running")

    def write_replay_manifest(
        self,
        entry: PlannedRun,
        configuration: RuntimeConfiguration,
    ) -> None:
        """Persist the resolved run declaration and reproducibility metadata."""
        directory = self.run_directory(entry)
        tracking_directory = directory / "tracking"
        tracking_directory.mkdir(parents=True, exist_ok=True)
        resolved = _json_value(asdict(configuration))
        resolved["experiment"]["seed"] = entry.seed
        resolved["pipeline"] = _json_value(entry.variant.pipeline)
        serialized = json.dumps(
            resolved,
            indent=2,
            sort_keys=True,
        )
        (directory / "resolved_config.json").write_text(
            serialized,
            encoding="utf-8",
        )
        manifest = {
            "schema_version": CURRENT_RESULT_SCHEMA_VERSION,
            "run_id": entry.run_id,
            "variant_id": entry.variant.variant_id,
            "seed": entry.seed,
            "configuration_sha256": hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "torch_deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            **_source_state(),
            "replay_contract": (
                "Replay local enumeration from stored inputs and RNG stream; "
                "verify ordered output sequence count and SHA-256 digest."
            ),
        }
        (tracking_directory / "replay_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_completed(
        self,
        entry: PlannedRun,
        result: OptimizationResult,
        snapshots: list[MemorySnapshot],
    ) -> None:
        """Persist candidates, stability measurements and completed status."""
        directory = self.run_directory(entry)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_candidates(directory / "candidates.csv", result)
        self._write_candidate_fields(directory / "fields.jsonl", result)
        torch.save(result.candidates.latent_origins.detach().cpu(), directory / "latent_origins.pt")
        tracking_directory = directory / "tracking"
        tracking_directory.mkdir(parents=True, exist_ok=True)
        self._write_stability(tracking_directory / "stability.csv", snapshots)
        self._write_status(
            entry,
            "completed",
            candidate_count=len(result.candidates),
            best_score=result.best_score,
            objective_name=result.objective_name,
            objective_direction=result.objective_direction,
        )

    def write_failed(self, entry: PlannedRun, error: BaseException) -> None:
        """Persist a failed status and concise exception information."""
        self._write_status(
            entry,
            "failed",
            error=f"{type(error).__name__}: {error}",
        )

    def _write_status(self, entry: PlannedRun, status: str, **values: Any) -> None:
        """Atomically replace one run status document."""
        directory = self.run_directory(entry)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CURRENT_RESULT_SCHEMA_VERSION,
            "status": status,
            "run_id": entry.run_id,
            "run_index": entry.index,
            "task_id": entry.task.task_id,
            "variant_id": entry.variant.variant_id,
            "seed": entry.seed,
            "sequence": entry.task.sequence,
            **values,
        }
        target = directory / "result.json"
        temporary = directory / "result.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(target)

    @staticmethod
    def _write_candidates(path: Path, result: OptimizationResult) -> None:
        """Write final candidate sequences and every scalar oracle score."""
        score_fields = {}
        for name, field in result.candidates.fields.items():
            if name.startswith("oracle.") and name.endswith(".score"):
                values = getattr(field, "values", None)
                if values is not None and values.ndim == 1:
                    score_fields[name] = values
        fieldnames = ("candidate_index", "sequence", *sorted(score_fields))
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for index, sequence in enumerate(result.candidates.sequences):
                writer.writerow(
                    {
                        "candidate_index": index,
                        "sequence": sequence,
                        **{
                            name: float(values[index].item())
                            for name, values in score_fields.items()
                        },
                    }
                )

    @staticmethod
    def _write_candidate_fields(path: Path, result: OptimizationResult) -> None:
        """Write final non-latent candidate fields as one JSON object per row."""
        from pep_compass.data.optimization import (
            ObjectField,
            OptionalField,
            SharedField,
            TensorField,
        )

        with path.open("w", encoding="utf-8") as stream:
            for index in range(len(result.candidates)):
                fields = {}
                for name, value in result.candidates.fields.items():
                    if isinstance(value, TensorField):
                        fields[name] = _json_value(
                            value.values[index].detach().cpu().tolist()
                        )
                    elif isinstance(value, ObjectField):
                        fields[name] = repr(value.values[index])
                    elif isinstance(value, SharedField):
                        fields[name] = _json_value(value.value)
                    elif isinstance(value, OptionalField):
                        fields[name] = {
                            "valid": bool(value.valid[index].item())
                        }
                stream.write(
                    json.dumps(
                        {"candidate_index": index, "fields": fields},
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    @staticmethod
    def _write_stability(path: Path, snapshots: list[MemorySnapshot]) -> None:
        """Write low-overhead runtime memory measurements."""
        fields = tuple(MemorySnapshot.__dataclass_fields__)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for snapshot in snapshots:
                writer.writerow({field: getattr(snapshot, field) for field in fields})


def _json_value(value: Any) -> Any:
    """Convert nested runtime values into deterministic JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _source_state() -> dict[str, Any]:
    """Return repository revision metadata without making replay depend on Git."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"source_revision": revision, "source_dirty": dirty}
