"""Shared bootstrap and subsampling engine for all experiment analyses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ClusterSampler:
    """Sample complete clusters while preserving rows inside each cluster."""

    columns: tuple[str, ...]
    sample_size: int | None = None
    replace: bool = True

    def sample(self, frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Return rows belonging to sampled cluster identifiers."""
        grouped = frame.groupby(list(self.columns), sort=False).groups
        group_keys = list(grouped)
        if not group_keys:
            return frame.iloc[0:0]
        size = self.sample_size or len(group_keys)
        if not self.replace and size > len(group_keys):
            raise ValueError("sample_size exceeds available clusters")
        selected = rng.choice(len(group_keys), size=size, replace=self.replace)
        pieces = [
            frame.loc[grouped[group_keys[index]]]
            for index in selected
        ]
        return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0]


@dataclass(frozen=True, slots=True)
class RowSampler:
    """Sample individual rows with or without replacement."""

    sample_size: int | None = None
    replace: bool = True

    def sample(self, frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Return a row-resampled frame."""
        size = self.sample_size or len(frame)
        indices = rng.choice(len(frame), size=size, replace=self.replace)
        return frame.iloc[indices].reset_index(drop=True)


class BootstrapEngine:
    """Evaluate arbitrary statistics under one shared resampling implementation."""

    def run(
        self,
        frame: pd.DataFrame,
        statistic: Callable[[pd.DataFrame], float | Sequence[float]],
        sampler: ClusterSampler | RowSampler,
        repetitions: int = 500,
        confidence: float = 0.95,
        random_seed: int = 0,
    ) -> pd.DataFrame:
        """Return bootstrap samples and percentile confidence bounds.

        :param frame: Source observations.
        :param statistic: Scalar or vector statistic evaluated on every sample.
        :param sampler: Shared row or cluster sampling policy.
        :param repetitions: Number of resampled statistics.
        :param confidence: Percentile interval coverage.
        :param random_seed: Deterministic generator seed.
        :return: Long-form samples with interval columns attached.
        """
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        rng = np.random.default_rng(random_seed)
        values = []
        for repetition in range(repetitions):
            value = np.atleast_1d(statistic(sampler.sample(frame, rng)))
            values.extend(
                {
                    "repetition": repetition,
                    "component": component,
                    "value": float(item),
                }
                for component, item in enumerate(value)
            )
        result = pd.DataFrame(values)
        tail = (1.0 - confidence) / 2.0
        intervals = result.groupby("component")["value"].agg(
            estimate="mean",
            confidence_low=lambda values: values.quantile(tail),
            confidence_high=lambda values: values.quantile(1.0 - tail),
        )
        return result.merge(intervals, on="component", how="left")
