"""Efficient process and CUDA memory sampling at execution boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from pep_compass.data.optimization import CandidateBatch
from pep_compass.optimization.stability_estimation.estimation import (
    estimate_batch_memory,
)
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

if TYPE_CHECKING:
    from pep_compass.optimization.engine.execution.state import OptimizationState


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Store one memory observation at a step boundary.

    ``rss_bytes`` and CUDA allocator counters are collected for every snapshot.
    ``batch_bytes`` requires traversal of Python-backed batch values and is
    therefore populated only at pipeline boundaries or configured progress
    thresholds.
    """

    label: str
    candidates: int
    batch_bytes: int | None
    rss_bytes: int | None
    cuda_allocated_bytes: int | None
    cuda_reserved_bytes: int | None
    cuda_peak_bytes: int | None
    detailed: bool
    detail_trigger: str | None


class NullStabilityMonitor:
    """Disable resource sampling without branching in engine callers."""

    def sample(
        self,
        label: str,
        batch: CandidateBatch,
        state: "OptimizationState | None" = None,
    ) -> MemorySnapshot | None:
        """Return no snapshot for a disabled monitor."""
        return None


class StabilityMonitor:
    """Sample and log process and CUDA memory at every step boundary.

    Cheap process and allocator counters are always recorded. Exact batch
    inspection is additionally performed at pipeline boundaries and whenever
    iterations, oracle calls, or generated candidates cross another configured
    fraction of their budget. A single snapshot reports the progress counters
    responsible for triggering detailed inspection.

    :param enabled: Disable all sampling while retaining the same interface.
    :param log_level: Logging level used for emitted snapshots.
    :param detailed_fraction: Fraction of a progress budget between detailed
        batch inspections; ``0.1`` produces ten checkpoints.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        log_level: int = 10,
        detailed_fraction: float = 0.1,
        detailed_every_sample: bool = False,
    ) -> None:
        if not 0 < detailed_fraction <= 1:
            raise ValueError("Detailed sampling fraction must be in (0, 1].")
        self.enabled = enabled
        self.log_level = log_level
        self.detailed_fraction = detailed_fraction
        self.detailed_every_sample = detailed_every_sample
        self.snapshots: list[MemorySnapshot] = []
        self._budgets: dict[str, int] = {}
        self._next_thresholds: dict[str, int] = {}

    def configure_budgets(
        self,
        *,
        iterations: int | None = None,
        oracle_calls: int | None = None,
        generated_candidates: int | None = None,
    ) -> None:
        """Configure progress budgets that trigger detailed batch inspection.

        :param iterations: Declared maximum engine iterations.
        :param oracle_calls: Optional oracle-call safety limit.
        :param generated_candidates: Optional candidate-generation safety limit.
        """
        values = {
            "iterations": iterations,
            "oracle_calls": oracle_calls,
            "generated_candidates": generated_candidates,
        }
        self._budgets = {
            name: value
            for name, value in values.items()
            if value is not None and value > 0
        }
        self._next_thresholds = {
            name: self._interval(value) for name, value in self._budgets.items()
        }

    def sample(
        self,
        label: str,
        batch: CandidateBatch,
        state: "OptimizationState | None" = None,
    ) -> MemorySnapshot | None:
        """Collect and log one memory snapshot.

        :param label: Stable execution-boundary label.
        :param batch: Candidate batch live at the boundary.
        :return: Collected snapshot, or ``None`` when monitoring is disabled.
        """
        if not self.enabled:
            return None
        triggers = self._consume_triggers(state)
        boundary_trigger = label in {"pipeline.input", "pipeline.output"}
        detailed = self.detailed_every_sample or boundary_trigger or bool(triggers)
        trigger = ",".join(triggers) if triggers else (
            "test_run" if self.detailed_every_sample else (
                "pipeline_boundary" if boundary_trigger else None
            )
        )
        estimate = estimate_batch_memory(batch) if detailed else None
        snapshot = MemorySnapshot(
            label=label,
            candidates=len(batch),
            batch_bytes=estimate.total_bytes if estimate is not None else None,
            rss_bytes=_current_rss_bytes(),
            cuda_allocated_bytes=_cuda_value(torch.cuda.memory_allocated),
            cuda_reserved_bytes=_cuda_value(torch.cuda.memory_reserved),
            cuda_peak_bytes=_cuda_value(torch.cuda.max_memory_allocated),
            detailed=detailed,
            detail_trigger=trigger,
        )
        self.snapshots.append(snapshot)
        logger.log(
            self.log_level,
            "Stability label=%s candidates=%s detailed=%s detail_trigger=%s "
            "batch_bytes=%s rss_bytes=%s cuda_allocated_bytes=%s "
            "cuda_reserved_bytes=%s cuda_peak_bytes=%s.",
            snapshot.label,
            snapshot.candidates,
            snapshot.detailed,
            snapshot.detail_trigger,
            snapshot.batch_bytes,
            snapshot.rss_bytes,
            snapshot.cuda_allocated_bytes,
            snapshot.cuda_reserved_bytes,
            snapshot.cuda_peak_bytes,
        )
        return snapshot

    def _interval(self, budget: int) -> int:
        """Return the number of progress units between detailed samples."""
        return max(1, math.ceil(budget * self.detailed_fraction))

    def _consume_triggers(
        self,
        state: "OptimizationState | None",
    ) -> tuple[str, ...]:
        """Advance crossed progress thresholds and return their names."""
        if state is None:
            return ()
        progress = {
            "iterations": state.completed_iterations,
            "oracle_calls": state.oracle_calls,
            "generated_candidates": state.generated_candidates,
        }
        triggered = []
        for name, budget in self._budgets.items():
            threshold = self._next_thresholds[name]
            current = progress[name]
            if current < threshold:
                continue
            triggered.append(
                f"{name}:{current}/{budget}"
            )
            interval = self._interval(budget)
            self._next_thresholds[name] = (
                (current // interval) + 1
            ) * interval
        return tuple(triggered)


def _current_rss_bytes() -> int | None:
    """Read current resident memory on Linux without an external dependency."""
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    try:
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, ValueError):
        return None


def _cuda_value(function) -> int | None:
    """Return one CUDA allocator counter without forcing device creation."""
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return int(function())
