"""Public executable PepCompass pipeline."""

from __future__ import annotations

import numpy as np
import torch

from pep_compass.autoencoder.base import Autoencoder
from pep_compass.data.optimization import (
    Candidate,
    CandidateBatch,
    SharedField,
    TensorField,
)
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.result import OptimizationResult
from pep_compass.optimization.stability_estimation.monitoring import (
    NullStabilityMonitor,
    StabilityMonitor,
)
from pep_compass.optimization.engine.execution.state import OptimizationLimits, OptimizationState
from pep_compass.optimization.engine.execution.step import Step
from pep_compass.optimization.tracking import NullStepTracker, StepTracker
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class PepCompassPipeline:
    """Execute a fully constructed PepCompass computation.

    The pipeline is independent of YAML, experiment planning, persistence and
    CLI concerns. Callers may construct it manually or obtain it from
    :class:`pep_compass.core.builder.PipelineBuilder`.

    :param autoencoder: Autoencoder shared by computational components.
    :param root: Root operation of the executable computation graph.
    :param tracker: Optional detailed step tracker.
    :param limits: Optional run-level computation limits.
    :param stability_monitor: Optional process and accelerator memory monitor.
    """

    def __init__(
        self,
        *,
        autoencoder: Autoencoder,
        root: Step,
        tracker: StepTracker | None = None,
        limits: OptimizationLimits | None = None,
        stability_monitor: StabilityMonitor | NullStabilityMonitor | None = None,
    ) -> None:
        self.autoencoder = autoencoder
        self.root = root
        self.tracker = tracker or NullStepTracker()
        self.limits = limits or OptimizationLimits()
        self.stability_monitor = stability_monitor or NullStabilityMonitor()

    def run(
        self,
        sequences: list[str],
        *,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Encode input sequences and execute the computation graph.

        :param sequences: Non-empty starting peptide sequences.
        :param seed: Optional deterministic run seed.
        :return: Final candidates and optional objective summary.
        :raises ValueError: If no starting sequence is supplied.
        """
        # Sanity check: Empty sequences 
        if not sequences:
            raise ValueError("At least one starting sequence is required.")

        # Setting seed 
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Initial candidate representation
        with torch.no_grad():
            latent_origins = self.autoencoder.encode_peptides(sequences)  # (B, D)
        context = OptimizationContext(
            autoencoder=self.autoencoder,
            tracker=self.tracker,
            seed=seed,
            rng=np.random.default_rng(seed),
            state=OptimizationState(limits=self.limits),
            stability_monitor=self.stability_monitor,
        )
        candidate_ids = torch.as_tensor(
            context.state.next_candidate_ids(len(sequences)),
            dtype=torch.long,
            device=latent_origins.device,
        )  # (B,)
        batch = CandidateBatch(sequences, latent_origins)
        batch = batch.with_field("lineage.candidate_id", TensorField(candidate_ids))
        batch = batch.with_field(
            "lineage.parent_candidate_id",
            TensorField(torch.full_like(candidate_ids, -1)),
        )

        # Computation lifecycle
        try:
            self.stability_monitor.sample("pipeline.input", batch, context.state)
            logger.info("Precomputing PepCompass pipeline steps.")
            self.root.precompute(context)
            logger.info(
                "Executing PepCompass pipeline for %s starting sequences.",
                len(sequences),
            )
            result = self.root(batch, context)
            self.stability_monitor.sample("pipeline.output", result, context.state)
            return self._summarize(result, context.state)
        finally:
            self.tracker.close()

    @staticmethod
    def _summarize(
        batch: CandidateBatch,
        state: OptimizationState,
    ) -> OptimizationResult:
        """Return the complete evaluated archive for the terminal objective."""
        score_names = [
            name
            for name, value in batch.fields.items()
            if name.startswith("oracle.")
            and name.endswith(".score")
            and isinstance(value, TensorField)
        ]
        if not score_names or len(batch) == 0:
            return OptimizationResult(candidates=batch)
        score_name = score_names[-1]
        prefix = score_name.removesuffix(".score")
        objective = prefix.removeprefix("oracle.")
        score_field = batch.fields[score_name]
        direction_field = batch.fields.get(f"{prefix}.direction")
        name_field = batch.fields.get(f"{prefix}.name")
        direction = (
            direction_field.value
            if isinstance(direction_field, SharedField)
            else "minimize"
        )
        scores = score_field.values
        if scores.ndim != 1:
            return OptimizationResult(candidates=batch)

        # Evaluated-candidate archive
        ## Oracle observations are the durable experiment result. The batch
        ## propagated by the loop contains only the latest acquisition batch.
        observations = state.observations.get(objective, {})
        archived_latents = state.observation_latents.get(objective, {})
        if observations and all(sequence in archived_latents for sequence in observations):
            sequences = tuple(observations)
            scores = torch.tensor(
                list(observations.values()),
                dtype=torch.float32,
            )  # (N,)
            latent_origins = torch.stack(
                [archived_latents[sequence] for sequence in sequences]
            )  # (N, D)
            batch = CandidateBatch(sequences, latent_origins).with_field(
                score_name,
                TensorField(scores),
            )
            batch = batch.with_field(
                f"{prefix}.direction",
                SharedField(direction),
            ).with_field(
                f"{prefix}.name",
                SharedField(objective),
            )
        best_index = int(
            scores.argmax().item()
            if direction == "maximize"
            else scores.argmin().item()
        )
        return OptimizationResult(
            candidates=batch,
            best_candidate=Candidate(
                batch.sequences[best_index],
                batch.latent_origins[best_index],
            ),
            best_score=float(scores[best_index].item()),
            objective_name=(
                str(name_field.value)
                if isinstance(name_field, SharedField)
                else objective
            ),
            objective_direction=direction,
        )
