"""Bayesian ROBOT selector implemented as a state-aware candidate filter."""

from __future__ import annotations

from cachetools import LRUCache
import numpy as np
import torch
import Levenshtein
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from pep_compass.optimization.components.filters.direct.base import DirectFilter
from pep_compass.optimization.components.filters.manager import FilterManager
from pep_compass.optimization.components.filters.direct.optimization.helpers.fingerprints import Map4Fingerprint
from pep_compass.optimization.components.filters.direct.optimization.helpers.kernel import (
    TanimotoSimilarityKernel,
)
from pep_compass.data.optimization import CandidateBatch, TensorField
from pep_compass.optimization.engine.execution.context import OptimizationContext


@FilterManager.register("robot")
class RobotSelector(DirectFilter):
    """Select a diverse batch using GP expected improvement.

    The selector reads observations collected by an oracle from
    :class:`OptimizationState`. With fewer than two observations it performs a
    deterministic seeded random initialization instead of fitting a GP.
    """

    def __init__(
        self,
        *,
        objective: str,
        batch_size: int = 1,
        maximize: bool = False,
        diversity_threshold: int = 2,
        acquisition_batch_size: int = 64,
        standardize: bool = False,
        device: str = "cpu",
    ) -> None:
        if batch_size < 1:
            raise ValueError("ROBOT batch size must be positive.")
        self.objective = objective
        self.batch_size = batch_size
        self.maximize = maximize
        self.diversity_threshold = diversity_threshold
        self.acquisition_batch_size = acquisition_batch_size
        self.standardize = standardize
        self.device = device
        self.fingerprint = Map4Fingerprint(input_type="fasta", chiral=False)
        self.cache = LRUCache(maxsize=5_000_000)

    def _execute(
        self,
        batch: CandidateBatch,
        context: OptimizationContext,
    ) -> CandidateBatch:
        observations = context.state.observations.get(self.objective, {})
        available_indices = [
            index
            for index, sequence in enumerate(batch.sequences)
            if sequence not in observations
        ]
        if not available_indices:
            context.state.stop_requested = True
            return batch.select([])
        available = batch.select(available_indices)
        if len(observations) < 2:
            count = min(self.batch_size, len(available))
            selected = context.rng.choice(len(available), size=count, replace=False)
            return available.select(selected.tolist())

        train_sequences = list(observations)
        train_scores = list(observations.values())
        train_x = torch.as_tensor(
            self._features(train_sequences),
            dtype=torch.float64,
            device=self.device,
        )
        train_y = torch.as_tensor(
            train_scores,
            dtype=torch.float64,
            device=self.device,
        ).reshape(-1, 1)
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            covar_module=TanimotoSimilarityKernel(),
            outcome_transform=Standardize(m=1) if self.standardize else None,
        ).to(self.device)
        likelihood = ExactMarginalLogLikelihood(model.likelihood, model).to(self.device)
        fit_gpytorch_mll(likelihood)
        model.eval()
        acquisition = LogExpectedImprovement(
            model=model,
            best_f=train_y.max() if self.maximize else train_y.min(),
            maximize=self.maximize,
        ).to(self.device)
        test_x = torch.as_tensor(
            self._features(list(available.sequences)),
            dtype=torch.float64,
            device=self.device,
        ).unsqueeze(1)
        scores = self._score_batches(acquisition, test_x)
        scored = available.with_field(
            "filter.robot.acquisition",
            TensorField(scores.to(available.latent_origins.device)),
        )
        indices = self._diverse_top_indices(available.sequences, scores)
        return scored.select(indices)

    def _features(self, sequences: list[str]) -> np.ndarray:
        unseen = [sequence for sequence in sequences if sequence not in self.cache]
        if unseen:
            for sequence, feature in zip(unseen, self.fingerprint(unseen)):
                self.cache[sequence] = feature
        return np.asarray([self.cache[sequence] for sequence in sequences])

    def _score_batches(
        self,
        acquisition: LogExpectedImprovement,
        features: torch.Tensor,
    ) -> torch.Tensor:
        values = []
        with torch.no_grad():
            for start in range(0, len(features), self.acquisition_batch_size):
                batch = features[start : start + self.acquisition_batch_size]
                duplicate = len(batch) == 1
                if duplicate:
                    batch = batch.repeat(2, 1, 1)
                scores = acquisition(batch)
                values.append(scores[:1] if duplicate else scores)
        return torch.cat(values)

    def _diverse_top_indices(
        self,
        sequences: tuple[str, ...],
        scores: torch.Tensor,
    ) -> list[int]:
        remaining = list(range(len(sequences)))
        selected: list[int] = []
        cpu_scores = scores.detach().cpu()
        while remaining and len(selected) < self.batch_size:
            best = max(remaining, key=lambda index: float(cpu_scores[index]))
            selected.append(best)
            remaining = [
                index
                for index in remaining
                if index != best
                and Levenshtein.distance(sequences[index], sequences[best])
                > self.diversity_threshold
            ]
        return selected
