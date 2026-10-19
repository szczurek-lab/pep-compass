"""Dispatch frozen jobs and execute each computation through the public wrapper."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from pep_compass.core.wrapper.pipeline import Pipeline
from pep_compass.optimization.saving.reader import AnalysisReader
from pep_compass.orchestrator.planning import save_plan
from pep_compass.utils.configuration import construct
from pep_compass.utils.logger import get_custom_logger

if TYPE_CHECKING:
    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.autoencoder.resource import AutoencoderResource
    from pep_compass.orchestrator.planning import ExecutionPlan, PlannedRun


logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class RunExecution:
    """Compact terminal status; full retained results remain in the shared reader."""

    index: int
    job_id: str
    status: str
    path: str | None
    run_id: str | None = None
    summary: dict | None = None
    error: str | None = None


class Orchestrator:
    """Coordinate independent Pipeline runs using directly configured execution strategies.

    :param autoencoder: Optional initialized resource, local execution only.
    """

    def __init__(
        self, *, autoencoder: AutoencoderResource | Autoencoder | None = None
    ) -> None:
        """Initialize this object with the declared configuration.

        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: AutoencoderResource | Autoencoder | None
        :rtype: None
        """
        self.autoencoder = autoencoder

    def run(self, plan: ExecutionPlan) -> tuple[RunExecution, ...] | Path:
        """Persist the plan and dispatch it through the selected backend.

        :param plan: A model-free validated ExecutionPlan.
        :return: Ordered RunExecution records, or a generated Slurm script path.

        :type plan: ExecutionPlan
        :rtype: tuple[RunExecution, ...] | Path
        """
        plan, path = save_plan(plan)
        backend = construct(plan.backend)
        return backend.run(self, plan, path)

    def run_entry(
        self,
        plan: ExecutionPlan,
        entry: PlannedRun,
        *,
        skip_completed: bool | None = None,
    ) -> RunExecution:
        """Execute or validate one job with exclusive output ownership.

        :param plan: Frozen experiment plan.
        :param entry: One job from that plan.
        :param skip_completed: Optional launch-policy override.
        :return: Terminal status. Existing incomplete output is never overwritten.

        :type plan: ExecutionPlan
        :type entry: PlannedRun
        :type skip_completed: bool | None
        :rtype: RunExecution
        """
        # Acquire exclusive ownership before examining or creating output.
        root = Path(plan.output_directory)
        lock = root / ".locks" / (entry.job_id + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        acquired = False
        result = None
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            acquired = True
            with os.fdopen(descriptor, "w") as stream:
                stream.write(str(os.getpid()))
            # Validate completed results or execute the public wrapper.
            skip = (
                plan.options.skip_completed
                if skip_completed is None
                else skip_completed
            )
            if entry.path is not None and Path(entry.path).exists():
                if not skip:
                    raise FileExistsError(
                        f"Output already exists: {entry.path}; enable skip_completed to validate it."
                    )
                record = self._completed(plan, entry)
            else:
                logger.info("Executing job=%s index=%s.", entry.job_id, entry.index)
                from pep_compass.utils.device import (
                    inspect_torch_runtime,
                    validate_execution_device,
                )

                device = entry.settings.execution.device or (
                    plan.schema.autoencoder.get("device", "cpu")
                    if plan.schema.autoencoder
                    else "cpu"
                )
                validate_execution_device(device, inspect_torch_runtime())
                pipeline = Pipeline.from_schema(
                    plan.schema, autoencoder=self.autoencoder
                )
                pipeline.set_runner_settings(**entry.settings.to_dict())
                pipeline.run(entry.sequences)
                result = pipeline.last_result
                record = RunExecution(
                    entry.index,
                    entry.job_id,
                    "completed",
                    entry.path,
                    result.reader.settings["run_id"],
                    result.summary,
                )
        except Exception as error:
            logger.error("Orchestrated job=%s failed.", entry.job_id, exc_info=True)
            record = RunExecution(
                entry.index,
                entry.job_id,
                "failed",
                entry.path,
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            if result is not None:
                result.close()
            if acquired:
                lock.unlink()
        # Persist orchestration status without modifying the computational bundle.
        if acquired:
            target = root / "statuses" / (entry.job_id + ".json")
            target.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(asdict(record), stream, indent=2)
            try:
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return record

    def _completed(self, plan: ExecutionPlan, entry: PlannedRun) -> bool:
        """Completed.

        :param plan: Plan.
        :type plan: ExecutionPlan
        :param entry: Entry.
        :type entry: PlannedRun
        :return: Result of this operation.
        :rtype: bool
        """
        reader = AnalysisReader(entry.path)
        try:
            manifest, summary = reader.settings, reader.summary
            if (
                summary["status"] != "completed"
                or manifest["schema"] != plan.schema.to_dict()
                or manifest["inputs"] != list(entry.sequences)
                or manifest["runner_settings"] != entry.settings.to_dict()
                or manifest["reproducibility"]["source_sha256"] != plan.source_sha256
            ):
                raise ValueError(
                    "Existing result does not match this completed computation."
                )
            if self.autoencoder is not None:
                from pep_compass.optimization.engine.runner import model_digest

                digest = model_digest(self.autoencoder)
                if (
                    digest is None
                    or manifest["reproducibility"]["autoencoder_sha256"] != digest
                ):
                    raise ValueError(
                        "Injected autoencoder identity cannot be verified for skipping."
                    )
            # REMARK: A changed model file invalidates skip even when its name is unchanged.
            for model in manifest["models"]:
                for artifact in model["files"]:
                    digest = hashlib.sha256()
                    with Path(artifact["path"]).open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != artifact["sha256"]:
                        raise ValueError(
                            "Existing result model artifacts have changed."
                        )
            return RunExecution(
                entry.index,
                entry.job_id,
                "skipped",
                entry.path,
                manifest["run_id"],
                summary,
            )
        finally:
            reader.close()
