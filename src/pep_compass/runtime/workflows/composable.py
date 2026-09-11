"""Composable optimization workflow backed by the core pipeline builder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pep_compass.autoencoder.strategies  # noqa: F401
from pep_compass.autoencoder.factory import AutoencoderFactory
from pep_compass.core.builder import PipelineBuilder
from pep_compass.optimization.pipeline import PepCompassPipeline
from pep_compass.optimization.stability_estimation.monitoring import StabilityMonitor
from pep_compass.optimization.tracking import StepTracker
from pep_compass.runtime.configuration.pipeline import parse_pipeline_specification
from pep_compass.runtime.configuration.schema import AutoencoderConfiguration


class ComposableWorkflow:
    """Build composable pipelines while reusing one immutable autoencoder."""

    def __init__(self, configuration: AutoencoderConfiguration) -> None:
        self.autoencoder = AutoencoderFactory.build(
            configuration.method,
            model=configuration.model,
            device=configuration.device,
            **dict(configuration.parameters),
        )
        self.builder = PipelineBuilder(self.autoencoder)

    def build_pipeline(
        self,
        pipeline_configuration: Mapping[str, Any],
        *,
        tracker: StepTracker,
        stability_monitor: StabilityMonitor,
    ) -> PepCompassPipeline:
        """Parse and construct one concrete pipeline variant."""
        specification = parse_pipeline_specification(pipeline_configuration)
        return self.builder.build(
            specification,
            tracker=tracker,
            stability_monitor=stability_monitor,
            initial_candidates=1,
        )
