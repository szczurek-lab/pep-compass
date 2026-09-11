"""Adapter exposing existing POLI black boxes as optimization steps."""

from __future__ import annotations

import numpy as np
import torch

from pep_compass.data.optimization import CandidateBatch, SharedField, TensorField
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.components.oracles.base import Oracle


class BlackBoxOracle(Oracle):
    """Evaluate sequences with an existing black box and attach its scores."""

    def __init__(
        self, black_box, *, field_name: str, batch_size: int | None = None
    ) -> None:
        self.black_box = black_box
        self.field_name = field_name
        self.batch_size = batch_size

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        remaining = context.state.remaining_oracle_calls()
        if remaining == 0:
            context.state.stop_requested = True
            return batch.select([])
        evaluated_batch = batch if remaining is None else batch.select(range(min(len(batch), remaining)))
        if len(evaluated_batch) == 0:
            scores = torch.empty(
                0,
                dtype=torch.get_default_dtype(),
                device=evaluated_batch.latent_origins.device,
            )
            return self._attach_result(evaluated_batch, scores)
        if self.batch_size is None:
            raw = self.black_box(np.asarray(evaluated_batch.sequences))
        else:
            values = []
            for start in range(0, len(evaluated_batch), self.batch_size):
                values.append(
                    self.black_box(
                        np.asarray(
                            evaluated_batch.sequences[start : start + self.batch_size]
                        )
                    )
                )
            raw = np.concatenate(values, axis=0)
        scores = torch.as_tensor(raw, device=evaluated_batch.latent_origins.device)
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        context.state.record_observations(
            self.field_name.removesuffix(".score").removeprefix("oracle."),
            evaluated_batch.sequences,
            [float(score) for score in scores.detach().cpu().tolist()],
            evaluated_batch.latent_origins,
        )
        return self._attach_result(evaluated_batch, scores)

    def _attach_result(
        self,
        batch: CandidateBatch,
        scores: torch.Tensor,
    ) -> CandidateBatch:
        """Attach score and oracle metadata fields to an evaluated batch."""
        result = batch.with_field(self.field_name, TensorField(scores))
        direction = (
            "maximize" if getattr(self.black_box, "maximize", False) else "minimize"
        )
        prefix = self.field_name.removesuffix(".score")
        result = result.with_field(f"{prefix}.direction", SharedField(direction))
        return result.with_field(
            f"{prefix}.name", SharedField(prefix.removeprefix("oracle."))
        )
