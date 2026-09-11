"""Composable trust-region selection and center-update policies."""

from __future__ import annotations

import Levenshtein
import torch

from pep_compass.optimization.components.filters.direct.base import DirectFilter
from pep_compass.optimization.components.filters.manager import FilterManager
from pep_compass.data.optimization import CandidateBatch, SharedField, TensorField
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.state import TrustRegionState


@FilterManager.register("trust_region")
class TrustRegionSelector(DirectFilter):
    """Retain candidates within the current objective-specific trust region."""

    def __init__(
        self,
        *,
        objective: str,
        geometry: str = "sequence",
        initial_radius: float,
    ) -> None:
        if geometry not in {"sequence", "latent"}:
            raise ValueError("Trust-region geometry must be sequence or latent.")
        if initial_radius <= 0:
            raise ValueError("Trust-region initial_radius must be positive.")
        self.objective = objective
        self.geometry = geometry
        self.initial_radius = initial_radius

    def _execute(self, batch: CandidateBatch, context: OptimizationContext) -> CandidateBatch:
        region = context.state.trust_regions.setdefault(
            self.objective, TrustRegionState(radius=self.initial_radius)
        )
        if region.center_sequence is None or region.center_latent is None:
            return batch.with_field(
                f"trust_region.{self.objective}.radius", SharedField(region.radius)
            )
        if self.geometry == "sequence":
            retained = [
                index
                for index, sequence in enumerate(batch.sequences)
                if Levenshtein.distance(sequence, region.center_sequence) <= region.radius
            ]
        else:
            distances = torch.linalg.vector_norm(
                batch.latent_origins - region.center_latent.to(batch.latent_origins.device),
                dim=1,
            )
            retained = torch.nonzero(distances <= region.radius, as_tuple=False).flatten().tolist()
        selected = batch.select(retained)
        return selected.with_field(
            f"trust_region.{self.objective}.radius", SharedField(region.radius)
        )


@FilterManager.register("trust_region_update")
class TrustRegionUpdater(DirectFilter):
    """Update center and radius from scores already attached by an oracle."""

    def __init__(
        self,
        *,
        objective: str,
        initial_radius: float,
        minimum_radius: float,
        maximum_radius: float,
        expand_factor: float = 2.0,
        shrink_factor: float = 0.5,
        success_tolerance: int = 3,
        failure_tolerance: int = 3,
        maximize: bool = False,
    ) -> None:
        if not 0 < minimum_radius <= initial_radius <= maximum_radius:
            raise ValueError("Trust-region radii must satisfy 0 < min <= initial <= max.")
        if expand_factor <= 1 or not 0 < shrink_factor < 1:
            raise ValueError("Trust-region expansion must exceed 1 and shrinkage be in (0, 1).")
        if success_tolerance < 1 or failure_tolerance < 1:
            raise ValueError("Trust-region tolerances must be positive.")
        self.objective = objective
        self.initial_radius = initial_radius
        self.minimum_radius = minimum_radius
        self.maximum_radius = maximum_radius
        self.expand_factor = expand_factor
        self.shrink_factor = shrink_factor
        self.success_tolerance = success_tolerance
        self.failure_tolerance = failure_tolerance
        self.maximize = maximize

    def _execute(self, batch: CandidateBatch, context: OptimizationContext) -> CandidateBatch:
        score_field = batch.fields.get(f"oracle.{self.objective}.score")
        if not isinstance(score_field, TensorField) or score_field.values.ndim != 1:
            raise ValueError(
                f"Trust-region update requires one-dimensional oracle.{self.objective}.score."
            )
        if len(batch) == 0:
            return batch
        scores = score_field.values
        best_index = int((scores.argmax() if self.maximize else scores.argmin()).item())
        score = float(scores[best_index].item())
        region = context.state.trust_regions.setdefault(
            self.objective, TrustRegionState(radius=self.initial_radius)
        )
        improved = region.best_score is None or (
            score > region.best_score if self.maximize else score < region.best_score
        )
        if improved:
            region.best_score = score
            region.center_sequence = batch.sequences[best_index]
            region.center_latent = batch.latent_origins[best_index].detach().clone()
            region.successes += 1
            region.failures = 0
            if region.successes >= self.success_tolerance:
                region.radius = min(region.radius * self.expand_factor, self.maximum_radius)
                region.successes = 0
        else:
            region.failures += 1
            region.successes = 0
            if region.failures >= self.failure_tolerance:
                region.radius = max(region.radius * self.shrink_factor, self.minimum_radius)
                region.failures = 0
        result = batch.with_field(
            f"trust_region.{self.objective}.radius", SharedField(region.radius)
        )
        return result.with_field(
            f"trust_region.{self.objective}.best_score", SharedField(region.best_score)
        )
