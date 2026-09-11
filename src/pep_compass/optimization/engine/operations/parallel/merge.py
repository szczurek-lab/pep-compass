"""Merge policies used exclusively after ``Parallel`` branch execution."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pep_compass.data.optimization import CandidateBatch

MergeMethod = Literal["concatenate", "interleave", "select_best", "weighted_sample"]


class BatchMerger(ABC):
    """Merge outputs from logically parallel optimization branches."""

    @abstractmethod
    def __call__(self, batches: Sequence[CandidateBatch]) -> CandidateBatch:
        """Return one batch containing merged branch outputs."""


class ConcatenateMerger(BatchMerger):
    """Append branch outputs without implicit selection or deduplication."""

    def __call__(self, batches: Sequence[CandidateBatch]) -> CandidateBatch:
        """Concatenate batches in configured branch order."""
        return CandidateBatch.concatenate(batches)


def _not_implemented(method: str) -> Callable[[Sequence[CandidateBatch]], CandidateBatch]:
    """Return a merge policy that reports an explicitly unsupported method."""
    def merge(batches: Sequence[CandidateBatch]) -> CandidateBatch:
        raise NotImplementedError(f"The {method} merge policy is not implemented yet.")

    return merge


@dataclass(frozen=True)
class FunctionMerger(BatchMerger):
    """Expose a merge function through the common merger interface."""

    function: Callable[[Sequence[CandidateBatch]], CandidateBatch]

    def __call__(self, batches: Sequence[CandidateBatch]) -> CandidateBatch:
        """Delegate merging to the configured function."""
        return self.function(batches)


def build_merger(method: MergeMethod) -> BatchMerger:
    """Build a branch merger from its configuration name."""
    if method == "concatenate":
        return ConcatenateMerger()
    if method in {"interleave", "select_best", "weighted_sample"}:
        return FunctionMerger(_not_implemented(method))
    raise ValueError(f"Unsupported merge method: {method}")
