"""Deterministic predictor and internal-stage fixtures for evaluation contracts."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from pep_compass.steps.base import Step
from pep_compass.steps.oracles.strategies.black_box import BlackBoxOracle


class CountingPredictor:
    """Record physical predictor inputs and return exact float32 length scores."""

    calls: ClassVar[list[tuple[str, ...]]] = []
    maximize = True

    def __init__(self, *, offset=0.0, **kwargs):
        self.offset = offset

    def __call__(self, sequences):
        self.calls.append(tuple(str(sequence) for sequence in sequences))
        return np.array(
            [[len(sequence) + self.offset] for sequence in sequences], dtype=np.float32
        )


@dataclass
class CountingOracle(BlackBoxOracle):
    """Use the real oracle cache with a small transparent predictor."""

    implementation = "tests.fixtures.evaluation_contracts.CountingPredictor"
    objective = "length"
    offset: float = 0.0


class InternalStages(Step):
    """Emit repeated stage intervals to verify per-invocation aggregation."""

    def _execute(self, batch, context):
        for index in range(len(batch)):
            with context.measure_stage("row_work", input_count=1, output_count=1):
                pass
        return batch


class ReverseDuplicates(Step):
    """Select a known occurrence order when identities and sequences repeat."""

    def _execute(self, batch, context):
        return (
            batch.select([1, 0, 1]).with_field(
                "marker", batch.fields["marker"].select([1, 0, 1])
            )
            if "marker" in batch.fields
            else batch.select([1, 0, 1])
        )


class PreparationStages(Step):
    """Use the common detailed-stage API during static preparation."""

    def precompute(self, context):
        with context.measure_stage("prepare_resource"):
            pass

    def _execute(self, batch, context):
        return batch
