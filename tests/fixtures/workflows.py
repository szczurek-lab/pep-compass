"""Runtime workflows that avoid external models in tests."""

from collections.abc import Mapping
from typing import Any

from pep_compass.optimization.engine.operations.flow import Flow
from pep_compass.optimization.pipeline import PepCompassPipeline
from tests.fixtures.autoencoders import MockAutoencoder
from tests.fixtures.components import SuffixStep


class MockWorkflow:
    """Build a deterministic one-step pipeline for runtime tests."""

    def build_pipeline(
        self,
        pipeline_configuration: Mapping[str, Any],
        *,
        tracker,
        stability_monitor,
    ) -> PepCompassPipeline:
        """Construct a pipeline using a suffix from mock configuration."""
        return PepCompassPipeline(
            autoencoder=MockAutoencoder(),
            root=Flow([SuffixStep(str(pipeline_configuration.get("suffix", "X")))]),
            tracker=tracker,
            stability_monitor=stability_monitor,
        )
