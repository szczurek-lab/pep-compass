"""Contract for runtime-selectable PepCompass workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pep_compass.optimization.pipeline import PepCompassPipeline
from pep_compass.optimization.stability_estimation.monitoring import StabilityMonitor
from pep_compass.optimization.tracking import StepTracker


class RuntimeWorkflow(Protocol):
    """Build an executable pipeline for one concrete plan variant."""

    def build_pipeline(
        self,
        pipeline_configuration: Mapping[str, Any],
        *,
        tracker: StepTracker,
        stability_monitor: StabilityMonitor,
    ) -> PepCompassPipeline:
        """Build one executable pipeline for a plan entry."""
