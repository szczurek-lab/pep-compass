"""Resolve model-free experiment plans and persist their immutable job identities."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from pep_compass.core.builder import PipelineBuilder
from pep_compass.data_schemas.execution import RunnerSettings
from pep_compass.data_schemas.graph import PipelineSchema
from pep_compass.optimization.saving.provenance import source_digest
from pep_compass.orchestrator.configuration import (
    OrchestrationSettings,
    load_plugins,
    plugin_fingerprints,
)
from pep_compass.orchestrator.declarations import create_pipeline
from pep_compass.utils.configuration import construct, locate, plain
from pep_compass.utils.logger import get_custom_logger
from pep_compass.utils.validation import validate_integer, validate_keys

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from omegaconf import DictConfig

    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.autoencoder.resource import AutoencoderResource


logger = get_custom_logger(__name__)


def _json(value: object) -> str:
    """Json.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: str
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: object) -> str:
    """Sha.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: str
    """
    return hashlib.sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class PlannedRun:
    """Independent wrapper invocation with resolved settings and stable identity."""

    index: int
    job_id: str
    sequences: tuple[str, ...]
    settings: RunnerSettings
    path: str | None


@dataclass(frozen=True)
class ExecutionPlan:
    """Frozen schema, input batches and launch policy; no live model objects."""

    fingerprint: str
    configuration: dict
    schema: PipelineSchema
    entries: tuple[PlannedRun, ...]
    options: OrchestrationSettings
    backend: dict
    working_directory: str
    output_directory: str
    source_sha256: str
    plugin_sha256: dict[str, str]

    @property
    def path(self) -> Path:
        """Return the immutable plan artifact path.

        :return: Result of this operation.
        :rtype: Path
        """
        return Path(self.output_directory) / "plans" / (self.fingerprint + ".json")

    def to_dict(self) -> dict[str, object]:
        """Return the portable, versioned JSON plan payload.

        :return: Result of this operation.
        :rtype: dict[str, object]
        """
        return {
            "format_version": 1,
            "fingerprint": self.fingerprint,
            "configuration": self.configuration,
            "schema": self.schema.to_dict(),
            "entries": [
                {**asdict(entry), "settings": entry.settings.to_dict()}
                for entry in self.entries
            ],
            "options": asdict(self.options),
            "backend": self.backend,
            "working_directory": self.working_directory,
            "output_directory": self.output_directory,
            "source_sha256": self.source_sha256,
            "plugin_sha256": self.plugin_sha256,
        }


def _inputs(
    configuration: Mapping[str, object], working_directory: Path
) -> tuple[tuple[str, ...], ...]:
    """Inputs.

    :param configuration: Resolved configuration for this operation.
    :type configuration: Mapping[str, object]
    :param working_directory: Working directory.
    :type working_directory: Path
    :return: Result of this operation.
    :rtype: tuple[tuple[str, ...], ...]
    """
    validate_keys(
        configuration, {"sequences", "path", "column", "mode", "limit"}, "input"
    )
    supplied = configuration.get("sequences")
    path = configuration.get("path")
    if (supplied is None) == (path is None):
        raise ValueError("Select exactly one of input.sequences or input.path.")
    if path is not None:
        source = Path(path)
        if not source.is_absolute():
            source = working_directory / source
        if source.suffix.lower() == ".csv":
            with source.open(newline="", encoding="utf-8") as stream:
                supplied = [
                    row[configuration.get("column", "sequence")]
                    for row in csv.DictReader(stream)
                ]
        else:
            text = source.read_text(encoding="utf-8")
            if text.lstrip().startswith(">"):
                supplied, fragments = [], []
                for line in text.splitlines():
                    if line.startswith(">"):
                        if fragments:
                            supplied.append("".join(fragments))
                            fragments = []
                    elif line.strip():
                        fragments.append(line.strip())
                if fragments:
                    supplied.append("".join(fragments))
            else:
                supplied = [line.strip() for line in text.splitlines() if line.strip()]
    if (
        not isinstance(supplied, (list, tuple))
        or not supplied
        or any(not isinstance(item, str) or not item for item in supplied)
    ):
        raise ValueError(
            "input must provide a non-empty collection of peptide strings."
        )
    limit = configuration.get("limit")
    validate_integer(limit, "input.limit", 1, optional=True)
    sequences = tuple(supplied[:limit])
    mode = configuration.get("mode", "batch")
    if mode not in {"batch", "individual"}:
        raise ValueError("input.mode must be batch or individual.")
    return (sequences,) if mode == "batch" else tuple((item,) for item in sequences)


def _identity(
    configuration: Mapping[str, object],
    batches: Sequence[Sequence[str]],
    source: str,
    working_directory: str | Path,
    plugin_sha256: Mapping[str, str],
) -> str:
    # REMARK: Dispatch and skip/failure policy do not alter computation identity.
    """Identity.

    :param configuration: Resolved configuration for this operation.
    :type configuration: Mapping[str, object]
    :param batches: Batches.
    :type batches: Sequence[Sequence[str]]
    :param source: Source.
    :type source: str
    :param working_directory: Working directory.
    :type working_directory: str | Path
    :param plugin_sha256: Plugin sha256.
    :type plugin_sha256: Mapping[str, str]
    :return: Result of this operation.
    :rtype: str
    """
    config = json.loads(_json(configuration))
    config.pop("backend", None)
    config["orchestrator"] = {
        key: config.get("orchestrator", {}).get(key, default)
        for key, default in (("name", "experiment"), ("output_dir", "results"))
    }
    return _sha(
        {
            "configuration": config,
            "batches": batches,
            "source": source,
            "plugins": plugin_sha256,
            "working_directory": str(working_directory),
        }
    )


def plan_configuration(
    configuration: DictConfig | Mapping[str, object],
    *,
    working_directory: str | Path | None = None,
    autoencoder: AutoencoderResource | Autoencoder | None = None,
) -> ExecutionPlan:
    """Validate and freeze a wrapper computation without loading models.

    :param configuration: Hydra DictConfig or a plain configuration mapping.
    :param working_directory: Base for input/template/output paths.
    :param autoencoder: Optional injected Python autoencoder, local dispatch only.
    :return: Validated ExecutionPlan with resolved schema and runner settings.
    :raises ValueError: If graph, inputs or configuration contracts are invalid.

    :type configuration: DictConfig | Mapping[str, object]
    :type working_directory: str | Path | None
    :type autoencoder: AutoencoderResource | Autoencoder | None
    :rtype: ExecutionPlan
    """
    # Resolve configuration and validate declarations without model construction.
    values = (
        OmegaConf.to_container(configuration, resolve=True, throw_on_missing=True)
        if OmegaConf.is_config(configuration)
        else json.loads(_json(configuration))
    )
    validate_keys(
        values,
        {"pipeline", "input", "runner", "orchestrator", "backend", "plugins"},
        "configuration",
    )
    load_plugins(values.get("plugins", []))
    working = Path(working_directory or Path.cwd()).resolve()
    options = OrchestrationSettings(**values.get("orchestrator", {}))
    root = Path(options.output_dir)
    output = ((working / root) if not root.is_absolute() else root) / options.name
    patch = values.get("runner", {})
    validate_keys(patch, {"execution", "storage", "results"}, "runner")
    pipeline = create_pipeline(
        values["pipeline"],
        autoencoder=autoencoder,
        execution=patch.get("execution"),
    )
    schema = pipeline.to_schema()
    if schema.autoencoder is None and autoencoder is None:
        raise ValueError("Declare a shared autoencoder before planning execution.")
    device = patch.get("execution", {}).get("device")
    if device is not None and schema.autoencoder is not None:
        schema = replace(
            schema, autoencoder={**plain(schema.autoencoder), "device": device}
        )
    PipelineBuilder(schema=schema, autoencoder=autoencoder).validate()
    # Freeze backend-owned resources separately from computational settings.
    backend = values["backend"]
    if "_target_" not in backend:
        raise ValueError("backend requires an explicit _target_.")
    target = locate(backend["_target_"])
    parameters = target.resolve_parameters(
        {key: value for key, value in backend.items() if key != "_target_"}, working
    )
    backend = {"_target_": backend["_target_"], **parameters}
    from pep_compass.orchestrator.backends.base import ExecutionBackend

    if not isinstance(construct(backend), ExecutionBackend):
        raise TypeError("backend._target_ must construct ExecutionBackend.")
    batches = _inputs(values["input"], working)
    source = source_digest()
    plugins = plugin_fingerprints(values.get("plugins", []))
    identity = _identity(values, batches, source, working, plugins)
    # Assign one independent result directory and RunnerSettings snapshot per job.
    entries = []
    for index, sequences in enumerate(batches):
        job_id = f"job_{index:05d}_{_sha([identity, index, sequences])[:20]}"
        storage = dict(patch.get("storage", {}))
        if storage.get("path") is not None:
            raise ValueError(
                "The orchestrator assigns storage.path separately to every job."
            )
        storage.setdefault("backend", "sqlite")
        storage["path"] = (
            str(output / "runs" / job_id) if storage["backend"] == "sqlite" else None
        )
        settings = pipeline.runner_settings.update(
            execution=patch.get("execution"),
            storage=storage,
            results=patch.get("results"),
        )
        entries.append(PlannedRun(index, job_id, sequences, settings, storage["path"]))
    logger.info(
        "Planned jobs=%s preset=%s.", len(entries), values["pipeline"].get("_target_")
    )
    return ExecutionPlan(
        identity,
        values,
        schema,
        tuple(entries),
        options,
        backend,
        str(working),
        str(output),
        source,
        plugins,
    )


def _write_plan_once(payload: bytes, target: Path) -> None:
    """Publish a complete JSON plan atomically without replacing existing files.

    :param payload: Payload.
    :type payload: bytes
    :param target: Target.
    :type target: Path
    :rtype: None
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(
            {"payload": payload, "sha256": _sha(payload)},
            stream,
            indent=2,
            allow_nan=False,
        )
    try:
        try:
            os.link(temporary, target)
            return True
        except FileExistsError:
            return False
    finally:
        temporary.unlink()


def save_plan(plan: ExecutionPlan) -> Path:
    """Freeze computation identities and the exact policy of each dispatch.

    :param plan: Validated plan.
    :return: Persisted plan and the exact launch artifact consumed by workers.
    :raises ValueError: If an existing artifact is corrupt or inconsistent.

    :type plan: ExecutionPlan
    :rtype: Path
    """
    target = plan.path
    payload = plan.to_dict()
    if _write_plan_once(payload, target):
        return plan, target
    persisted = load_plan(target)
    if persisted.fingerprint != plan.fingerprint:
        raise ValueError("Frozen plan identity differs from the requested computation.")
    # Preserve graph identities while recording updated dispatch policy separately.
    plan = replace(persisted, options=plan.options, backend=plan.backend)
    payload = plan.to_dict()
    if payload == persisted.to_dict():
        return plan, target
    target = target.with_name(target.stem + ".launch-" + _sha(payload)[:20] + ".json")
    if not _write_plan_once(payload, target):
        existing = load_plan(target)
        if existing.to_dict() != payload:
            raise ValueError(
                "Frozen launch artifact differs from the requested policy."
            )
    return plan, target


def load_plan(path: str | Path) -> ExecutionPlan:
    """Load and validate a frozen plan without constructing models.

    :param path: Versioned plan artifact.
    :return: Validated ExecutionPlan.
    :raises ValueError: If integrity, source code or schema validation fails.

    :type path: str | Path
    :rtype: ExecutionPlan
    """
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = envelope["payload"]
    if envelope["sha256"] != _sha(payload) or payload["format_version"] != 1:
        raise ValueError("Frozen plan integrity/version mismatch.")
    configuration = payload["configuration"]
    load_plugins(configuration.get("plugins", []))
    if payload["source_sha256"] != source_digest():
        raise ValueError(
            "Frozen plan source code differs from the current installation."
        )
    if payload["plugin_sha256"] != plugin_fingerprints(
        configuration.get("plugins", [])
    ):
        raise ValueError(
            "Frozen plan plugin source differs from the current installation."
        )
    entries = tuple(
        PlannedRun(
            entry["index"],
            entry["job_id"],
            tuple(entry["sequences"]),
            RunnerSettings().update(**entry["settings"]),
            entry["path"],
        )
        for entry in payload["entries"]
    )
    identity = _identity(
        configuration,
        tuple(entry.sequences for entry in entries),
        payload["source_sha256"],
        payload["working_directory"],
        payload["plugin_sha256"],
    )
    if identity != payload["fingerprint"]:
        raise ValueError("Frozen plan computation identity mismatch.")
    schema = PipelineSchema.from_dict(payload["schema"])
    from pep_compass.core.wrapper.pipeline import Pipeline

    pipeline = Pipeline.from_schema(schema)
    pipeline.validate()
    return ExecutionPlan(
        identity,
        configuration,
        schema,
        entries,
        OrchestrationSettings(**payload["options"]),
        payload["backend"],
        payload["working_directory"],
        payload["output_directory"],
        payload["source_sha256"],
        payload["plugin_sha256"],
    )
