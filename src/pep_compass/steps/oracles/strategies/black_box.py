"""Adapter exposing existing POLI black boxes as optimization steps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from pep_compass.data_schemas.candidates import CandidateBatch, SharedField, TensorField
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.steps.base import Step
from pep_compass.utils.logger import get_custom_logger

if TYPE_CHECKING:
    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.data_schemas.contracts import StepContract


logger = get_custom_logger(__name__)


@dataclass
class BlackBoxOracle(Step):
    """Evaluate a complete candidate batch and retain aligned objective scores.

    :param batch_size: Native predictor batch size; None uses that predictor's default.
    :param parallelize: Enable native black-box parallel processing, default False.
    :param num_workers: Optional native worker count.
    :param evaluation_budget: Optional native evaluation budget, reset for each run.
    :param force_isolation: Enable native black-box process isolation.
    :param evaluation_batch_size: Optional adapter microbatch size; all completed results are retained.
    :param model: Named weight variant owned by this model implementation.
    :param models_directory: Optional external directory replacing the packaged model root.
    :param device: Device used by this objective's predictor, independent of the autoencoder.
    """

    display_parameters = ("model", "device", "scale", "evaluation_batch_size")
    category = "oracle"

    batch_size: int | None = None
    parallelize: bool = False
    num_workers: int | None = None
    evaluation_budget: int | float | None = None
    force_isolation: bool = False
    evaluation_batch_size: int | None = None
    implementation: ClassVar[str]
    objective: ClassVar[str]

    def __post_init__(self) -> None:
        """Post init.

        :rtype: None
        """
        from pep_compass.utils.validation import validate_integer

        for name in ("batch_size", "num_workers", "evaluation_batch_size"):
            validate_integer(getattr(self, name), "oracle." + name, 1, optional=True)
        for name in ("parallelize", "force_isolation"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(name + " must be boolean.")
        if self.evaluation_budget is not None and (
            isinstance(self.evaluation_budget, bool)
            or not np.isfinite(self.evaluation_budget)
            or self.evaluation_budget < 0
        ):
            raise ValueError(
                "Oracle evaluation_budget must be finite and non-negative, or None."
            )
        self._configure(
            {
                field.name: getattr(self, field.name)
                for field in fields(self)
                if field.init
            }
        )
        self.field_name = "oracle." + self.objective + ".score"

    def _initialize(self, autoencoder: Autoencoder | None = None) -> None:
        """Initialize.

        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder | None
        :rtype: None
        """
        module, _, name = self.implementation.rpartition(".")
        values = self.configuration
        values.pop("evaluation_batch_size", None)
        if values.get("evaluation_budget") is None:
            values.pop("evaluation_budget", None)
        self.black_box = getattr(import_module(module), name)(**values)

    def contract(self) -> StepContract:
        """Contract.

        :return: Result of this operation.
        :rtype: StepContract
        """
        from pep_compass.data_schemas.contracts import StepContract

        return StepContract(
            produced_fields=tuple(
                "oracle." + self.objective + suffix
                for suffix in (".score", ".direction", ".name")
            )
        )

    def precompute(self, context: OptimizationContext) -> None:
        """Reset per-run evaluation state while retaining expensive model weights.

        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :rtype: None
        """
        identity = hashlib.sha256(
            json.dumps(self.to_config(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        context.state.bind_objective(self.objective, identity)
        for name in ("reset_evaluation_budget", "clear_cache"):
            reset = getattr(self.black_box, name, None)
            if callable(reset):
                reset()

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        # Run-local cache; first occurrences determine physical predictor input order.
        """Execute.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        cached = context.state.observations.setdefault(self.objective, {})
        unseen_indices = []
        pending = set()
        for index, sequence in enumerate(batch.sequences):
            if sequence not in cached and sequence not in pending:
                pending.add(sequence)
                unseen_indices.append(index)
        remaining = context.state.remaining_oracle_calls()
        if unseen_indices and remaining == 0:
            context.state.stop_requested = True
            batch = batch.select(
                [
                    index
                    for index, sequence in enumerate(batch.sequences)
                    if sequence in cached
                ]
            )
            unseen_indices = []
        if unseen_indices:
            evaluated = batch.select(unseen_indices)
            chunks = []
            chunk_size = self.evaluation_batch_size or len(evaluated)
            with context.measure_stage(
                "inference", input_count=len(evaluated), output_count=len(evaluated)
            ):
                for start in range(0, len(evaluated), chunk_size):
                    chunks.append(
                        self.black_box(
                            np.asarray(evaluated.sequences[start : start + chunk_size])
                        )
                    )
            raw = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
            scores = torch.as_tensor(
                raw, device=batch.latent_origins.device
            )  # (U,) or (U, 1)
            if scores.ndim == 2 and scores.shape[1] == 1:
                scores = scores[:, 0]  # (U,)
            if scores.ndim != 1 or scores.shape[0] != len(evaluated):
                raise ValueError(
                    "A scalar oracle must return one score per evaluated sequence."
                )
            if not scores.is_floating_point():
                scores = scores.to(torch.float64)  # (U,)
            context.state.objective_dtypes[self.objective] = str(
                scores.dtype
            ).removeprefix("torch.")
            # Soft budgets stop subsequent work; all new scores in this batch survive.
            context.state.record_observations(
                self.objective,
                evaluated.sequences,
                [float(score) for score in scores.detach().cpu().tolist()],
                direction="maximize"
                if getattr(self.black_box, "maximize", False)
                else "minimize",
                lineage={
                    name: value.values.detach().cpu().tolist()
                    for name, value in evaluated.fields.items()
                    if name.startswith("lineage.") and isinstance(value, TensorField)
                },
            )
        if len(batch):
            context.state.last_objective = self.objective
        dtype = getattr(
            torch, context.state.objective_dtypes.get(self.objective, "float64")
        )
        aligned = torch.tensor(
            [cached[sequence] for sequence in batch.sequences],
            dtype=dtype,
            device=batch.latent_origins.device,
        )  # (B,)
        logger.debug(
            "Oracle objective=%s new=%s reused=%s.",
            self.objective,
            len(unseen_indices),
            len(batch) - len(unseen_indices),
        )
        if context.saver is not None:
            context.saver.link_evaluated_candidates(batch, self.objective)
        return self._attach_result(batch, aligned)

    def _attach_result(
        self,
        batch: CandidateBatch,
        scores: torch.Tensor,
    ) -> CandidateBatch:
        """Attach score and oracle metadata fields to an evaluated batch.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param scores: Scalar scores or a score tensor aligned with the corresponding candidates.
        :type scores: torch.Tensor
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        result = batch.with_field(self.field_name, TensorField(scores))
        direction = (
            "maximize" if getattr(self.black_box, "maximize", False) else "minimize"
        )
        prefix = self.field_name.removesuffix(".score")
        result = result.with_field(f"{prefix}.direction", SharedField(direction))
        return result.with_field(
            f"{prefix}.name", SharedField(prefix.removeprefix("oracle."))
        )
