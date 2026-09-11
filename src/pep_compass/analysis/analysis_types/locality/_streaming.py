"""Bounded-memory numerical accumulators for locality analyses."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from time import monotonic

import numpy as np


def _duration(seconds: float | None) -> str:
    """Format a duration for compact progress output."""
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass(slots=True)
class ProgressReporter:
    """Print row-based batch progress, throughput, and estimated completion.

    :param label: Human-readable analysis stage.
    :param total: Total number of rows expected across all batches.
    :param every_batches: Emit progress after this many processed batches.
    :param enabled: Whether progress output is emitted.
    """

    label: str
    total: int
    every_batches: int = 1
    enabled: bool = True
    _processed: int = field(default=0, init=False)
    _batches: int = field(default=0, init=False)
    _started_at: float = field(default_factory=monotonic, init=False)

    def __post_init__(self) -> None:
        if self.every_batches < 1:
            raise ValueError("Progress interval must be positive")
        if self.enabled:
            print(
                f"[{self.label}] Starting: {self.total:,} rows to process",
                flush=True,
            )

    def update(self, rows: int) -> None:
        """Record one completed batch and optionally print its progress."""
        self._processed += rows
        self._batches += 1
        if not self.enabled:
            return
        if self._batches % self.every_batches and self._processed < self.total:
            return
        elapsed = monotonic() - self._started_at
        rate = self._processed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - self._processed)
        eta = remaining / rate if rate > 0 else None
        percentage = (
            min(100.0, 100.0 * self._processed / self.total)
            if self.total
            else 100.0
        )
        print(
            f"[{self.label}] batch={self._batches:,} "
            f"rows={self._processed:,}/{self.total:,} ({percentage:.1f}%) "
            f"rate={rate:,.0f} rows/s elapsed={_duration(elapsed)} "
            f"ETA={_duration(eta)}",
            flush=True,
        )

    @property
    def processed(self) -> int:
        """Return the number of rows reported as completed."""
        return self._processed

    @property
    def elapsed_seconds(self) -> float:
        """Return wall-clock seconds since reporter initialization."""
        return monotonic() - self._started_at


@dataclass(slots=True)
class NestedProgress:
    """Track one overall bar across stages, plus a resettable current-stage bar.

    Each :meth:`start_stage` call replaces the current-stage reporter; the
    previous one has already printed its own completion line on its last
    :meth:`update` (a :class:`ProgressReporter` reaches 100% on the update
    that meets its ``total`), so no explicit finalize step is needed.

    :param label: Human-readable computation name.
    :param total: Total rows expected across every stage.
    :param enabled: Whether progress output is emitted.
    """

    label: str
    total: int
    enabled: bool = True
    _overall: ProgressReporter = field(init=False)
    _current: ProgressReporter | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._overall = ProgressReporter(
            f"{self.label}:overall", self.total, enabled=self.enabled
        )

    def start_stage(self, stage_label: str, stage_total: int) -> None:
        """Start a new current-stage bar, e.g. one per processed run."""
        self._current = ProgressReporter(
            f"{self.label}:{stage_label}", stage_total, enabled=self.enabled
        )

    def update(self, rows: int) -> None:
        """Advance both the current-stage bar and the overall bar."""
        if self._current is not None:
            self._current.update(rows)
        self._overall.update(rows)


@dataclass(slots=True)
class RunningMoments:
    """Accumulate finite values without retaining individual observations."""

    count: int = 0
    total: float = 0.0

    def update(self, values: np.ndarray) -> None:
        """Add finite values from one processing chunk."""
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        self.count += int(finite.size)
        self.total += float(finite.sum())

    @property
    def mean(self) -> float:
        """Return the accumulated arithmetic mean."""
        return self.total / self.count if self.count else float("nan")


@dataclass(slots=True)
class Reservoir:
    """Maintain a uniform bounded sample from a value stream.

    :param capacity: Maximum number of retained observations.
    :param seed: Seed controlling deterministic replacement.
    """

    capacity: int
    seed: int = 0
    _values: list[object] = field(default_factory=list, init=False)
    _seen: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("Reservoir capacity must be positive")
        self._rng = np.random.default_rng(self.seed)

    def update(self, values: Iterable[object]) -> None:
        """Consume observations using Algorithm R reservoir sampling."""
        for value in values:
            self._seen += 1
            if len(self._values) < self.capacity:
                self._values.append(value)
                continue
            replacement = int(self._rng.integers(0, self._seen))
            if replacement < self.capacity:
                self._values[replacement] = value

    @property
    def values(self) -> list[object]:
        """Return a copy of retained observations."""
        return list(self._values)

    @property
    def seen(self) -> int:
        """Return the total number of consumed observations."""
        return self._seen
