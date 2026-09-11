"""Tests for trajectory-isolated local enumeration."""

import torch
import pytest

from pep_compass.data.optimization import CandidateBatch
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.operations import Flow, LocalEnumeration
from pep_compass.optimization.pipeline import PepCompassPipeline
from tests.fixtures.autoencoders import MockAutoencoder
from tests.fixtures.components import SuffixStep


class RecordingWalker(SuffixStep):
    """Advance one-row trajectories and record input cardinality."""

    def __init__(self) -> None:
        super().__init__("W")
        self.input_sizes: list[int] = []

    def _execute(self, batch, context):
        self.input_sizes.append(len(batch))
        return super()._execute(batch, context)


class PairGenerator(SuffixStep):
    """Emit two local candidates for every trajectory point."""

    def __init__(self) -> None:
        super().__init__("")

    def _execute(self, batch, context):
        indices = torch.arange(len(batch), device=batch.latent_origins.device).repeat_interleave(2)
        expanded = batch.repeat_from_parents(indices)
        return expanded.with_sequences(
            [f"{sequence}{suffix}" for sequence in batch.sequences for suffix in ("A", "B")]
        )


class RandomWalker(SuffixStep):
    """Expose Torch RNG draws in sequences for reproducibility assertions."""

    def __init__(self) -> None:
        super().__init__("")

    def _execute(self, batch, context):
        draws = torch.randint(0, 1_000_000, (len(batch),))  # (B,)
        return batch.with_sequences(
            [
                f"{sequence}:{int(draw)}"
                for sequence, draw in zip(batch.sequences, draws)
            ]
        )


def test_local_enumeration_does_not_feed_mutations_back_to_walker() -> None:
    """Every SORBES call must receive only its private trajectory point."""
    walker = RecordingWalker()
    operation = LocalEnumeration(
        walker=walker,
        mutation_generator=PairGenerator(),
        filters=Flow([]),
        trajectories=2,
        trajectory_execution="sequential",
        iterations=2,
        walk_time=None,
    )
    batch = CandidateBatch(["S"], torch.zeros((1, 2)))  # (B=1, D=2)

    result = operation(batch, OptimizationContext(autoencoder=MockAutoencoder()))

    assert walker.input_sizes == [1, 1, 1, 1]
    # Two trajectories, two SORBES points, and three emissions per point:
    # the walk point plus two locally generated candidates.
    assert len(result) == 12
    assert result.sequences.count("SWA") == 2
    assert result.sequences.count("SWWB") == 2


def test_batched_local_enumeration_advances_all_trajectories_together() -> None:
    """Batched execution must replace per-trajectory model calls with batch calls."""
    walker = RecordingWalker()
    operation = LocalEnumeration(
        walker=walker,
        mutation_generator=PairGenerator(),
        filters=Flow([]),
        trajectories=3,
        trajectory_execution="batched",
        iterations=2,
        walk_time=None,
        include_walk_points=False,
    )
    batch = CandidateBatch(["S"], torch.zeros((1, 2)))  # (B=1, D=2)

    result = operation(
        batch,
        OptimizationContext(autoencoder=MockAutoencoder(), seed=7),
    )

    assert walker.input_sizes == [3, 3]
    assert len(result) == 12
    assert result.sequences.count("SWA") == 3
    assert result.sequences.count("SWWB") == 3


def test_batched_local_enumeration_rejects_unknown_execution() -> None:
    """An execution typo must fail before any model is called."""
    with pytest.raises(ValueError, match="sequential or batched"):
        LocalEnumeration(
            walker=RecordingWalker(),
            mutation_generator=PairGenerator(),
            filters=Flow([]),
            trajectories=1,
            trajectory_execution="threads",
            iterations=1,
            walk_time=None,
        )


def test_batched_trajectories_are_independent_and_seed_reproducible() -> None:
    """A run seed must reproduce distinct random draws for batched trajectories."""
    operation = LocalEnumeration(
        walker=RandomWalker(),
        mutation_generator=SuffixStep("M"),
        filters=Flow([]),
        trajectories=4,
        trajectory_execution="batched",
        iterations=1,
        walk_time=None,
        include_walk_points=False,
    )
    pipeline = PepCompassPipeline(autoencoder=MockAutoencoder(), root=operation)

    first = pipeline.run(["S"], seed=1234).candidates.sequences
    second = pipeline.run(["S"], seed=1234).candidates.sequences
    different_seed = pipeline.run(["S"], seed=1235).candidates.sequences

    assert first == second
    assert first != different_seed
    assert len(set(first)) == 4
