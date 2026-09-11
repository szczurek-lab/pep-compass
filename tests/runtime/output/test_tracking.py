"""Tests for streamed runtime replay checkpoints."""

from __future__ import annotations

import csv

import torch

from pep_compass.data.optimization import CandidateBatch, ObjectField, TensorField
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.runtime.output.tracking import CSVStepTracker


class LocalEnumeration:
    """Minimal named step used to exercise tracker lifecycle storage."""

    name = "LocalEnumeration"


class SorbesWalker:
    """Minimal named step used to exercise trajectory checkpoint storage."""

    name = "SorbesWalker"


def test_tracker_streams_replay_shards_without_candidate_snapshots(tmp_path) -> None:
    """Replay state must be durable without retaining every candidate row."""
    tracker = CSVStepTracker(tmp_path, level="normal", candidate_snapshots="none")
    context = OptimizationContext(autoencoder=object(), seed=17)
    batch = CandidateBatch(["AA"], torch.tensor([[1.0, 2.0]])).with_field(
        "lineage.candidate_id", TensorField(torch.tensor([0]))
    ).with_field(
        "lineage.parent_candidate_id", TensorField(torch.tensor([-1]))
    )

    local_handle = tracker.begin_step(LocalEnumeration(), batch, context.scope, context)
    tracker.end_step(
        local_handle,
        LocalEnumeration(),
        batch,
        batch,
        context.scope,
        context,
    )
    point_batch = batch.with_field(
        "tracking.trajectory_id", ObjectField(["trajectory/0"])
    ).with_field(
        "tracking.trajectory_index", TensorField(torch.tensor([0]))
    ).with_field(
        "tracking.rng_stream_seed", TensorField(torch.tensor([17]))
    ).with_field(
        "tracking.trajectory_step", TensorField(torch.tensor([0]))
    ).with_field("point_id", ObjectField([1])).with_field(
        "walker.adjusted_time_step", TensorField(torch.tensor([0.1]))
    )
    point_handle = tracker.begin_step(SorbesWalker(), batch, context.scope, context)
    tracker.end_step(
        point_handle,
        SorbesWalker(),
        batch,
        point_batch,
        context.scope,
        context,
    )
    tracker.close()

    with (tmp_path / "local_enumerations.csv").open(newline="", encoding="utf-8") as stream:
        local_row = next(csv.DictReader(stream))
    with (tmp_path / "trajectory_points.csv").open(newline="", encoding="utf-8") as stream:
        point_row = next(csv.DictReader(stream))
    assert (tmp_path / local_row["checkpoint_file"]).is_file()
    assert (tmp_path / point_row["checkpoint_file"]).is_file()
    assert (tmp_path / "candidates.csv").read_text(encoding="utf-8").splitlines() == [
        "execution_id,candidate_index,sequence,latent_origin,fields"
    ]
