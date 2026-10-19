"""Replay retained LocalEnumeration boundaries without runtime configuration."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.state import OptimizationState
from pep_compass.optimization.saving.saver import sequence_digest
from pep_compass.utils.logger import get_custom_logger
from pep_compass.utils.randomness import random_scope

if TYPE_CHECKING:
    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.optimization.saving.reader import AnalysisReader


logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class ReplayResult:
    """Reconstructed candidates with an explicit sequence-integrity verdict."""

    candidates: object
    expected_count: int
    expected_sha256: str

    @property
    def matches(self) -> bool:
        """Compare ordered sequence content and count with the saved boundary.

        :return: Result of this operation.
        :rtype: bool
        """
        return (
            len(self.candidates) == self.expected_count
            and sequence_digest(self.candidates.sequences) == self.expected_sha256
        )


def reconstruct(
    reader: AnalysisReader,
    checkpoint_id: int,
    *,
    autoencoder: Autoencoder | None = None,
) -> ReplayResult:
    """Rebuild one named operation, restore state and verify its output.

    :param reader: Reader of a run with retained reconstruction boundaries.
    :param checkpoint_id: Stable persisted checkpoint identity.
    :param autoencoder: Required for originally injected, unnamed resources.
    :return: ReplayResult with a digest verdict; no oracle is rerun implicitly.

    :type reader: AnalysisReader
    :type checkpoint_id: int
    :type autoencoder: Autoencoder | None
    :rtype: ReplayResult
    """
    from pep_compass.core.builder import PipelineBuilder
    from pep_compass.data_schemas.graph import PipelineSchema
    from pep_compass.optimization.engine.inspection import find_step
    from pep_compass.optimization.engine.runner import model_digest
    from pep_compass.steps.flow.strategies.local_enumeration import (
        LocalEnumeration,
    )

    row = reader.backend.execute(
        "SELECT * FROM checkpoints WHERE checkpoint_id=?", (checkpoint_id,)
    ).fetchone()
    if row is None:
        raise KeyError(checkpoint_id)
    if row["phase"] != "before_prepare_call" or row["output_sha256"] is None:
        raise ValueError("Checkpoint has no completed, supported replay boundary.")
    checkpoint = reader._load_checkpoint(row["payload_id"])
    logger.info("Reconstructing checkpoint=%s node=%s.", checkpoint_id, row["node_id"])
    from pep_compass.optimization.saving.provenance import source_digest

    if reader.settings["reproducibility"]["source_sha256"] != source_digest():
        raise ValueError("Package sources differ from the recorded replay environment.")
    with random_scope(None, preserve_unseeded=True):
        schema = PipelineSchema.from_dict(reader.settings["schema"])
        target = next(
            (node for node, _, _ in schema.walk() if node.node_id == row["node_id"]),
            None,
        )
        if target is None:
            raise ValueError("Checkpoint refers to a missing graph declaration.")
        compiled = PipelineBuilder(
            schema=PipelineSchema((target,), autoencoder=schema.autoencoder),
            autoencoder=autoencoder,
        ).build(initial_fields=checkpoint["batch"].fields)
        expected = reader.settings["reproducibility"]["autoencoder_sha256"]
        if expected is not None and model_digest(compiled.autoencoder) != expected:
            raise ValueError("Autoencoder weights differ from the recorded run.")
        operation = find_step(compiled.root, row["node_id"])
        if not isinstance(operation, LocalEnumeration):
            raise ValueError("Checkpoint does not refer to a LocalEnumeration node.")
        state = OptimizationState(**checkpoint["state"])
        scope = checkpoint["scope"]
        from dataclasses import replace

        context = OptimizationContext(
            autoencoder=compiled.autoencoder,
            seed=checkpoint["seed"],
            state=state,
            step_state=checkpoint["step_state"],
        )
        # Restore the parent scope: Step.__call__ adds the operation's own name.
        context = replace(context, scope=replace(scope, path=scope.path[:-1]))
        operation.precompute(context)
        # Restore the boundary after preparing component-local static resources.
        ## REMARK: precompute may initialize per-run context values. Replaying a
        ## later boundary must retain its saved values rather than those defaults.
        checkpoint = reader._load_checkpoint(row["payload_id"])
        context = replace(
            context,
            state=OptimizationState(**checkpoint["state"]),
            step_state=checkpoint["step_state"],
        )
        saved = checkpoint["rng"]
        random.setstate(saved["python"])
        np.random.set_state(saved["numpy_global"])
        context.rng.bit_generator.state = saved["numpy_context"]
        torch.set_rng_state(saved["torch_cpu"])
        if saved["torch_cuda"] is not None:
            if not torch.cuda.is_available():
                raise ValueError(
                    "This checkpoint requires the recorded CUDA RNG state."
                )
            torch.cuda.set_rng_state_all(saved["torch_cuda"])
        torch.use_deterministic_algorithms(saved["deterministic"])
        torch.backends.cudnn.benchmark = saved["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = saved["cudnn_deterministic"]
        batch = checkpoint["batch"]
        device = getattr(compiled.autoencoder, "device", "cpu")
        from pep_compass.data_schemas.candidates import (
            CandidateBatch,
            OptionalField,
            TensorField,
        )

        fields = {}
        for name, field in batch.fields.items():
            if isinstance(field, TensorField):
                field = TensorField(field.values.to(device))
            elif isinstance(field, OptionalField) and isinstance(
                field.values, TensorField
            ):
                field = OptionalField(
                    TensorField(field.values.values.to(device)), field.valid.to(device)
                )
            fields[name] = field
        batch = CandidateBatch(batch.sequences, batch.latent_origins.to(device), fields)
        candidates = operation(batch, context)
        return ReplayResult(candidates, row["output_count"], row["output_sha256"])
