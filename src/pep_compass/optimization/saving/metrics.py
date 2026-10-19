"""Bounded process-RSS sampling and non-overlapping measurement scopes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import perf_counter

from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

DEFAULT_MEMORY_SAMPLE_INTERVAL_SECONDS = 0.01


def process_rss() -> int | None:
    """Read resident process bytes where the operating system exposes them.

    :return: Current process RSS, or None if unavailable.
    :rtype: int | None
    """
    try:
        with open("/proc/self/statm", encoding="ascii") as source:
            return int(source.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


@dataclass
class Measurement:
    """One active wall-clock interval and its sampled process-RSS maximum."""

    started: float
    memory_start_bytes: int | None
    memory_peak_bytes: int | None
    memory_end_bytes: int | None = None
    duration_seconds: float | None = None


class MemorySampler:
    """Sample active intervals without retaining an ever-growing time series.

    :param enabled: Whether timing and memory measurements are requested.
    :type enabled: bool
    :param interval_seconds: Positive sampling interval, recorded in the manifest.
    :type interval_seconds: float
    """

    def __init__(
        self,
        enabled: bool,
        interval_seconds: float = DEFAULT_MEMORY_SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param enabled: Whether this optional measurement or recording operation is enabled.
        :type enabled: bool
        :param interval_seconds: Interval in seconds between process-memory samples.
        :type interval_seconds: float
        :rtype: None
        """
        from pep_compass.utils.validation import validate_positive

        validate_positive(interval_seconds, "memory_sampling.interval_seconds")
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self._lock = Lock()
        self._active: dict[int, Measurement] = {}
        self._stop = Event()
        self._thread = None
        if enabled:
            logger.debug(
                "Starting process-RSS sampler interval_seconds=%s.", interval_seconds
            )
            self._thread = Thread(
                target=self._sample, name="pep-compass-rss", daemon=True
            )
            self._thread.start()

    def begin(self) -> Measurement | None:
        """Begin one measurement without changing computational randomness.

        :return: Mutable interval handle, or None when disabled.
        :rtype: Measurement | None
        """
        if not self.enabled:
            return None
        rss = process_rss()
        result = Measurement(perf_counter(), rss, rss)
        with self._lock:
            self._active[id(result)] = result
        return result

    def end(self, measurement: Measurement | None) -> Measurement | None:
        """Finish an interval, including its final endpoint in the sampled maximum.

        :param measurement: Handle returned by begin.
        :type measurement: Measurement | None
        :return: Completed handle, or None when disabled.
        :rtype: Measurement | None
        """
        if measurement is not None:
            with self._lock:
                self._update(measurement, process_rss())
                measurement.memory_end_bytes = process_rss()
                measurement.duration_seconds = perf_counter() - measurement.started
                self._active.pop(id(measurement), None)
        return measurement

    @staticmethod
    def _update(measurement: Measurement, rss: int | None) -> None:
        """Update.

        :param measurement: Measurement.
        :type measurement: Measurement
        :param rss: Rss.
        :type rss: int | None
        :rtype: None
        """
        if rss is not None:
            measurement.memory_peak_bytes = max(measurement.memory_peak_bytes or 0, rss)

    def _sample(self) -> None:
        """Sample.

        :rtype: None
        """
        while not self._stop.wait(self.interval_seconds):
            rss = process_rss()
            with self._lock:
                for measurement in self._active.values():
                    self._update(measurement, rss)

    def close(self) -> None:
        """Join the sampler before the run's storage ownership is transferred.

        :rtype: None
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
