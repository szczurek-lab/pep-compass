"""Typed access to replay checkpoints written by the optimization runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True, slots=True)
class ReplayVerification:
    """Report whether reconstructed local-enumeration output matches a checkpoint."""

    execution_id: int
    expected_count: int
    actual_count: int
    expected_sha256: str
    actual_sha256: str

    @property
    def matches(self) -> bool:
        """Return whether both ordered sequence count and digest match."""
        return (
            self.expected_count == self.actual_count
            and self.expected_sha256 == self.actual_sha256
        )


@dataclass(frozen=True, slots=True)
class LocalEnumerationReplay:
    """Materialized result of reconstructing one local-enumeration boundary.

    :param execution_id: Tracked LocalEnumeration lifecycle identity.
    :param candidates: Reconstructed candidates before downstream global steps.
    :param verification: Digest comparison against the original run.
    """

    execution_id: int
    candidates: Any
    verification: ReplayVerification


class RunReplay:
    """Read final results and minimal checkpoints for one materialized run."""

    def __init__(self, run_directory: str | Path) -> None:
        self.root = Path(run_directory)
        self.tracking = self.root / "tracking"
        if not (self.root / "result.json").is_file():
            raise FileNotFoundError(self.root / "result.json")

    @property
    def result(self) -> dict[str, Any]:
        """Return terminal run metadata."""
        return json.loads((self.root / "result.json").read_text(encoding="utf-8"))

    @property
    def manifest(self) -> dict[str, Any]:
        """Return runtime and replay-contract metadata."""
        path = self.tracking / "replay_manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def resolved_configuration(self) -> dict[str, Any]:
        """Return the exact resolved configuration persisted for this run."""
        return json.loads(
            (self.root / "resolved_config.json").read_text(encoding="utf-8")
        )

    @property
    def final_candidates(self) -> pd.DataFrame:
        """Return final sequences and all scalar oracle scores."""
        return pd.read_csv(self.root / "candidates.csv")

    @property
    def final_latents(self) -> torch.Tensor:
        """Return final candidate latent origins shaped ``(B, D)``."""
        return torch.load(
            self.root / "latent_origins.pt",
            map_location="cpu",
            weights_only=True,
        )

    @property
    def local_enumerations(self) -> pd.DataFrame:
        """Return local-enumeration replay boundaries and output digests."""
        return pd.read_csv(self.tracking / "local_enumerations.csv")

    @property
    def trajectory_points(self) -> pd.DataFrame:
        """Return SORBES point metadata indexed into :attr:`trajectory_latents`."""
        return pd.read_csv(self.tracking / "trajectory_points.csv")

    @property
    def trajectory_latents(self) -> torch.Tensor:
        """Return SORBES point latents shaped ``(N, D)`` on explicit request.

        New runs stream independent shards. This property intentionally joins
        them only when an analysis asks for the complete tensor; ordinary run
        tracking and reader construction do not materialize it.
        """
        legacy = self.tracking / "trajectory_latents.pt"
        if legacy.is_file():
            return torch.load(legacy, map_location="cpu", weights_only=True)
        points = self.trajectory_points
        if points.empty:
            return torch.empty((0, 0), dtype=torch.float32)
        relative_paths = [
            str(path)
            for path in dict.fromkeys(points["checkpoint_file"].dropna())
            if str(path)
        ]
        chunks = []
        for relative_path in relative_paths:
            chunks.append(
                torch.load(
                    self.tracking / relative_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )
        return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0))

    def trajectory(self, trajectory_id: str) -> tuple[pd.DataFrame, torch.Tensor]:
        """Return ordered metadata and latent points for one trajectory."""
        points = self.trajectory_points
        points = points[points["trajectory_id"] == trajectory_id].sort_values(
            ["trajectory_step", "execution_id"]
        )
        if "checkpoint_file" not in points.columns:
            indices = torch.as_tensor(points["latent_index"].to_numpy(), dtype=torch.long)
            return points.reset_index(drop=True), self.trajectory_latents.index_select(
                0, indices
            )
        shards: dict[str, torch.Tensor] = {}
        latents = []
        for _, row in points.iterrows():
            relative_path = str(row["checkpoint_file"])
            shard = shards.get(relative_path)
            if shard is None:
                shard = torch.load(
                    self.tracking / relative_path,
                    map_location="cpu",
                    weights_only=True,
                )
                shards[relative_path] = shard
            latents.append(shard[int(row["checkpoint_index"])])
        return points.reset_index(drop=True), torch.stack(latents)

    def local_enumeration_input(
        self,
        execution_id: int,
    ) -> tuple[tuple[str, ...], torch.Tensor]:
        """Return sequences and latent checkpoint used by one local enumeration."""
        rows = self.local_enumerations
        matched = rows[rows["execution_id"] == execution_id]
        if len(matched) != 1:
            raise KeyError(f"Unknown local-enumeration execution: {execution_id}")
        row = matched.iloc[0]
        if (
            "checkpoint_file" in row
            and pd.notna(row["checkpoint_file"])
            and str(row["checkpoint_file"])
        ):
            checkpoint = self.local_enumeration_checkpoint(execution_id)
            return tuple(checkpoint["sequences"]), checkpoint["latent_origins"]
        start = int(row["input_latent_start"])
        count = int(row["input_latent_count"])
        latents = torch.load(
            self.tracking / "local_enumeration_inputs.pt",
            map_location="cpu",
            weights_only=True,
        )
        sequences = tuple(json.loads(row["input_sequences"]))
        return sequences, latents[start : start + count]

    def local_enumeration_checkpoint(self, execution_id: int) -> dict[str, Any]:
        """Load the compact input and RNG checkpoint for one local enumeration.

        :param execution_id: LocalEnumeration execution identity.
        :return: CPU sequences, latents, lineage identifiers, and RNG states.
        :raises KeyError: If no unique checkpoint belongs to the execution.
        """
        rows = self.local_enumerations
        matched = rows[rows["execution_id"] == execution_id]
        if len(matched) != 1:
            raise KeyError(f"Unknown local-enumeration execution: {execution_id}")
        relative_value = matched.iloc[0]["checkpoint_file"]
        if pd.isna(relative_value) or not str(relative_value):
            raise ValueError("This legacy run has no standalone replay checkpoint.")
        relative_path = str(relative_value)
        return torch.load(
            self.tracking / relative_path,
            map_location="cpu",
            weights_only=False,
        )

    def reconstruct_local_enumeration(
        self,
        execution_id: int,
        *,
        device: str | None = None,
    ) -> LocalEnumerationReplay:
        """Rebuild one tracked LocalEnumeration without rerunning the whole run.

        :param execution_id: LocalEnumeration execution identity.
        :param device: Optional replay device; defaults to the recorded device.
        :return: Candidates and verification against the original output digest.
        :raises ValueError: If the recorded pipeline contains zero or multiple
            local-enumeration operations.

        REMARK: Exact equality requires the same package revision, model weights,
        device class, and deterministic-kernel settings as the original run.
        """
        from pep_compass.core.builder import PipelineBuilder
        from pep_compass.data.optimization import CandidateBatch, TensorField
        from pep_compass.optimization.engine.execution.context import OptimizationContext
        from pep_compass.optimization.engine.execution.state import OptimizationState
        from pep_compass.optimization.stability_estimation.monitoring import (
            NullStabilityMonitor,
        )
        from pep_compass.optimization.tracking import NullStepTracker
        from pep_compass.runtime.configuration.pipeline import parse_pipeline_specification
        from pep_compass.runtime.configuration.schema import AutoencoderConfiguration
        from pep_compass.runtime.workflows.composable import ComposableWorkflow

        checkpoint = self.local_enumeration_checkpoint(execution_id)
        configuration = self.resolved_configuration
        autoencoder_raw = configuration["autoencoder"]
        autoencoder_configuration = AutoencoderConfiguration(
            method=str(autoencoder_raw["method"]),
            model=str(autoencoder_raw["model"]),
            device=device or str(autoencoder_raw.get("device", "cpu")),
            parameters=dict(autoencoder_raw.get("parameters", {})),
        )
        workflow = ComposableWorkflow(autoencoder_configuration)
        specification = parse_pipeline_specification(configuration["pipeline"])
        pipeline = PipelineBuilder(workflow.autoencoder).build(specification)
        operations = _find_local_enumerations(pipeline.root)
        if len(operations) != 1:
            raise ValueError(
                "Replay currently requires exactly one LocalEnumeration operation; "
                f"found {len(operations)}."
            )

        # Restore the boundary RNG state before executing the isolated operation.
        np.random.set_state(checkpoint["numpy_global_rng_state"])
        torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
        if (
            device is not None
            and str(device).startswith("cuda")
            and "torch_cuda_rng_states" in checkpoint
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_states"])
        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint["numpy_rng_state"]
        latent_origins = checkpoint["latent_origins"].to(workflow.autoencoder.device)
        batch = CandidateBatch(checkpoint["sequences"], latent_origins)
        batch = batch.with_field(
            "lineage.candidate_id",
            TensorField(checkpoint["candidate_ids"].to(latent_origins.device)),
        ).with_field(
            "lineage.parent_candidate_id",
            TensorField(checkpoint["parent_candidate_ids"].to(latent_origins.device)),
        )
        state = OptimizationState(limits=specification.limits)
        if len(checkpoint["candidate_ids"]):
            state.created_candidates = int(checkpoint["candidate_ids"].max()) + 1
        context = OptimizationContext(
            autoencoder=workflow.autoencoder,
            tracker=NullStepTracker(),
            seed=checkpoint.get("seed"),
            rng=rng,
            state=state,
            stability_monitor=NullStabilityMonitor(),
        )
        operation = operations[0]
        operation.precompute(context)
        candidates = operation(batch, context)
        return LocalEnumerationReplay(
            execution_id=execution_id,
            candidates=candidates,
            verification=self.verify_local_enumeration(
                execution_id, candidates.sequences
            ),
        )

    def verify_local_enumeration(
        self,
        execution_id: int,
        reconstructed_sequences: Sequence[str],
    ) -> ReplayVerification:
        """Compare replayed ordered sequences with the stored result checkpoint."""
        rows = self.local_enumerations
        matched = rows[rows["execution_id"] == execution_id]
        if len(matched) != 1:
            raise KeyError(f"Unknown local-enumeration execution: {execution_id}")
        row = matched.iloc[0]
        actual = list(reconstructed_sequences)
        payload = json.dumps(
            actual,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return ReplayVerification(
            execution_id=execution_id,
            expected_count=int(row["output_count"]),
            actual_count=len(actual),
            expected_sha256=str(row["output_sequences_sha256"]),
            actual_sha256=hashlib.sha256(payload).hexdigest(),
        )


def _find_local_enumerations(root: Any) -> list[Any]:
    """Return LocalEnumeration nodes from an already constructed step tree."""
    from pep_compass.optimization.engine.operations.local_enumeration import (
        LocalEnumeration,
    )

    if isinstance(root, LocalEnumeration):
        return [root]
    children: list[Any] = []
    for attribute in ("steps", "body", "branches"):
        value = getattr(root, attribute, None)
        if isinstance(value, (list, tuple)):
            children.extend(value)
        elif isinstance(value, dict):
            children.extend(value.values())
        elif value is not None:
            children.append(value)
    result: list[Any] = []
    for child in children:
        result.extend(_find_local_enumerations(child))
    return result
