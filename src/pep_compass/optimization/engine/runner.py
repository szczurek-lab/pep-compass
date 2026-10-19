"""Execute a single constructed graph through the established pipeline engine."""

from __future__ import annotations

import hashlib
import platform
from typing import TYPE_CHECKING
from uuid import uuid4

import torch

from pep_compass.data_schemas.execution import OptimizationLimits, RunnerSettings
from pep_compass.data_schemas.results import RunResult
from pep_compass.optimization.engine.execution.context import OptimizationContext
from pep_compass.optimization.engine.execution.state import OptimizationState
from pep_compass.optimization.saving.backends import (
    SQLiteStorageBackend,
)
from pep_compass.optimization.saving.metrics import (
    DEFAULT_MEMORY_SAMPLE_INTERVAL_SECONDS,
)
from pep_compass.optimization.saving.provenance import source_digest
from pep_compass.optimization.saving.reader import AnalysisReader
from pep_compass.optimization.saving.saver import Saver
from pep_compass.utils.logger import get_custom_logger
from pep_compass.utils.randomness import random_scope

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pep_compass.data_schemas.graph import PipelineSchema
    from pep_compass.optimization.engine.pipeline import ExecutablePipeline


logger = get_custom_logger(__name__)


def model_digest(model: object) -> str | None:
    """Fingerprint tensor weights when an injected model exposes state_dict.

    :param model: Named model variant.
    :type model: object
    :return: Result of this operation.
    :rtype: str | None
    """
    if not hasattr(model, "state_dict"):
        return None
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Runner:
    """Own one run's RNG, observation retention and resource finalization.

    :param pipeline: Built ExecutablePipeline; operator implementations are reused.
    :param schema: Immutable public graph declaration.
    :param settings: Validated defaults for this runner.
    :param model_manifest: Named model artifact provenance collected at build.
    """

    def __init__(
        self,
        pipeline: ExecutablePipeline,
        schema: PipelineSchema,
        settings: RunnerSettings | None = None,
        model_manifest: Sequence[Mapping[str, object]] = (),
    ) -> None:
        """Initialize this object with the declared configuration.

        :param pipeline: Pipeline.
        :type pipeline: ExecutablePipeline
        :param schema: Immutable declaration or its explicitly supported serialized representation.
        :type schema: PipelineSchema
        :param settings: Validated settings or a resolved settings manifest.
        :type settings: RunnerSettings | None
        :param model_manifest: Model manifest.
        :type model_manifest: Sequence[Mapping[str, object]]
        :rtype: None
        """
        self.pipeline = pipeline
        self.schema = schema
        self.settings = settings or RunnerSettings()
        self.model_manifest = tuple(model_manifest)

    def run(
        self, sequences: Sequence[str], *, settings: RunnerSettings | None = None
    ) -> RunResult:
        """Return a RunResult; public return-mode selection belongs to the wrapper.

        :param sequences: Non-empty sequence collection.
        :param settings: Optional immutable settings override.
        :return: Completed run with a lazy reader.
        :raises Exception: Original computational or persistence failure.

        :type sequences: Sequence[str]
        :type settings: RunnerSettings | None
        :rtype: RunResult
        """
        settings = settings or self.settings
        from pep_compass.utils.validation import validate_sequences

        sequences = validate_sequences(sequences)
        if settings.execution.device is not None:
            actual = getattr(self.pipeline.autoencoder, "device", "cpu")
            if str(actual) != settings.execution.device:
                raise ValueError(
                    "Execution device differs from the built autoencoder; rebuild its resources."
                )
        if settings.storage.reconstruction == "none" and any(
            node.category == "save" for node, _, _ in self.schema.walk()
        ):
            raise ValueError(
                "Snapshots require reconstruction=boundaries or trajectories."
            )
        limits = OptimizationLimits(
            settings.execution.oracle_calls, settings.execution.generated_candidates
        )
        state = OptimizationState(limits=limits)
        manifest = {
            "format": "pep_compass.run",
            "format_version": 2,
            "memory_measurement": {
                "enabled": settings.storage.metrics_level != "none",
                "scope": "process_rss",
                "method": "sampled",
                "interval_seconds": DEFAULT_MEMORY_SAMPLE_INTERVAL_SECONDS,
            },
            "run_id": "run_" + uuid4().hex,
            "schema": self.schema.to_dict(),
            "inputs": list(sequences),
            "runner_settings": settings.to_dict(),
            "models": list(self.model_manifest),
            "reproducibility": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "rng_contract_version": 2,
                "source_sha256": source_digest(),
                "autoencoder_sha256": model_digest(self.pipeline.autoencoder),
            },
        }
        backend = SQLiteStorageBackend(manifest, settings.storage.path)
        saver = Saver(backend, settings.storage)
        state.observer = saver
        context = OptimizationContext(
            autoencoder=self.pipeline.autoencoder,
            tracker=saver,
            seed=settings.execution.seed,
            state=state,
            stability_monitor=self.pipeline.stability_monitor,
            saver=saver,
        )
        logger.info(
            "Executing graph run=%s backend=%s.",
            manifest["run_id"],
            settings.storage.backend,
        )
        try:
            with random_scope(settings.execution.seed):
                result = self.pipeline._run(
                    list(sequences),
                    seed=settings.execution.seed,
                    context=context,
                    close_tracker=False,
                )
            saver.finalize(result, state)
        except BaseException as error:
            logger.error("Graph run=%s failed.", manifest["run_id"], exc_info=True)
            try:
                saver.finalize(
                    None,
                    state,
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
            except Exception:
                logger.error("Could not persist the failed-run summary.", exc_info=True)
            try:
                backend.close()
            except Exception:
                logger.error("Could not close failed-run storage.", exc_info=True)
            raise
        reader = AnalysisReader(backend)
        return RunResult(reader.summary, reader, backend.path)
