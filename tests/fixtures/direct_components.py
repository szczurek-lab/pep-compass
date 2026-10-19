"""Deterministic analytical components for lifecycle and replay regression tests."""

import random

import torch

from pep_compass.data_schemas.candidates import CandidateBatch, SharedField, TensorField
from pep_compass.steps.base import Step


class LengthOracle(Step):
    """Evaluate deterministic sequence lengths through the real state contract."""

    def _execute(self, batch, context):
        scores = [float(len(sequence)) for sequence in batch.sequences]
        lineage = {
            name: field.values.tolist()
            for name, field in batch.fields.items()
            if name.startswith("lineage.") and isinstance(field, TensorField)
        }
        context.state.record_observations(
            "length",
            batch.sequences,
            scores,
            direction="maximize",
            lineage=lineage,
        )
        return batch.with_field(
            "oracle.length.score", TensorField(torch.tensor(scores))
        ).with_field("oracle.length.direction", SharedField("maximize"))


class FirstFilter(Step):
    def _execute(self, batch, context):
        return batch.select([0] if len(batch) else [])


class RandomWalker(Step):
    category = "walker"

    def _execute(self, batch, context):
        draws = context.rng.integers(0, 3, size=len(batch))
        sequences = [
            sequence + "ACD"[int(draw)]
            for sequence, draw in zip(batch.sequences, draws, strict=True)
        ]
        return CandidateBatch(
            sequences,
            batch.latent_origins + torch.rand_like(batch.latent_origins),
            {
                **batch.fields,
                "walker.adjusted_time_step": TensorField(
                    torch.full((len(batch),), 0.1)
                ),
            },
        )


class PairGenerator(Step):
    def _execute(self, batch, context):
        indices = torch.arange(len(batch)).repeat_interleave(2)  # (2 * B,)
        result = batch.repeat_from_parents(indices)
        sequences = [sequence + random.choice("GK") for sequence in result.sequences]
        result = result.with_sequences(sequences)
        context.state.record_generated_candidates(len(result))
        return result


class Failure(Step):
    """Raise in a selected lifecycle phase to verify durable failures."""

    def __init__(self, phase):
        self._configure({"phase": phase})
        self.phase = phase

    def precompute(self, context):
        if self.phase == "precompute":
            raise RuntimeError("expected precompute failure")

    def prepare_call(self, batch, context):
        if self.phase == "prepare_call":
            raise RuntimeError("expected prepare_call failure")

    def _execute(self, batch, context):
        raise RuntimeError("expected execute failure")
