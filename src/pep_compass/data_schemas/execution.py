"""Validated settings independent of the algorithm's declarative graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pep_compass.utils.validation import validate_integer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self


@dataclass
class OptimizationLimits:
    """Optional soft thresholds consumed at complete-batch boundaries."""

    oracle_calls: int | None = None
    generated_candidates: int | None = None

    def __post_init__(self) -> None:
        """Validate thresholds without coercing booleans or fractions.

        :rtype: None
        """
        validate_integer(self.oracle_calls, "limits.oracle_calls", optional=True)
        validate_integer(
            self.generated_candidates, "limits.generated_candidates", optional=True
        )


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    """Configure one run; loop iteration counts remain in the graph."""

    seed: int | None = None
    device: str | None = None
    oracle_calls: int | None = None
    generated_candidates: int | None = None
    max_workers: int = 1

    def __post_init__(self) -> None:
        """Post init.

        :rtype: None
        """
        validate_integer(self.seed, "execution.seed", optional=True)
        validate_integer(self.oracle_calls, "execution.oracle_calls", optional=True)
        validate_integer(
            self.generated_candidates, "execution.generated_candidates", optional=True
        )
        if isinstance(self.max_workers, bool) or self.max_workers != 1:
            raise ValueError(
                "A single-graph Runner requires max_workers=1; experiment dispatch is outside this API."
            )
        if self.device is not None and (
            not isinstance(self.device, str) or not self.device
        ):
            raise ValueError("execution.device must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Select retained information; buffering never samples the history."""

    backend: Literal["memory", "sqlite"] = "memory"
    path: str | Path | None = None
    flush_every: int = 1
    reconstruction: Literal["none", "boundaries", "trajectories"] = "none"
    component_data: Literal["none", "summary", "full"] = "none"
    metrics_level: Literal["none", "basic", "basic_and_components"] = "basic"
    save_lineage: bool = True

    def __post_init__(self) -> None:
        """Post init.

        :rtype: None
        """
        if self.backend not in {"memory", "sqlite"}:
            raise ValueError("storage.backend must be memory or sqlite.")
        if self.backend == "sqlite" and self.path is None:
            raise ValueError("SQLite storage requires a directory path.")
        if self.backend == "memory" and self.path is not None:
            raise ValueError(
                "Memory storage has no persistent path; use result.save(path)."
            )
        if self.path is not None:
            object.__setattr__(self, "path", str(Path(self.path)))
        validate_integer(self.flush_every, "storage.flush_every", 1)
        if self.reconstruction not in {"none", "boundaries", "trajectories"}:
            raise ValueError("Invalid reconstruction setting.")
        if self.component_data not in {"none", "summary", "full"}:
            raise ValueError("Invalid component_data setting.")
        if self.metrics_level not in {"none", "basic", "basic_and_components"}:
            raise ValueError("Invalid metrics_level setting.")
        if not isinstance(self.save_lineage, bool):
            raise ValueError("storage.save_lineage must be a boolean.")


@dataclass(frozen=True, slots=True)
class ResultsSettings:
    """Choose a view of the internally created RunResult."""

    return_mode: Literal["result", "reader", "path"] = "result"

    def __post_init__(self) -> None:
        """Post init.

        :rtype: None
        """
        if self.return_mode not in {"result", "reader", "path"}:
            raise ValueError("results.return_mode must be result, reader or path.")


@dataclass(frozen=True, slots=True)
class RunnerSettings:
    """Snapshot execution, retention and public result selection."""

    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    results: ResultsSettings = field(default_factory=ResultsSettings)

    def __post_init__(self) -> None:
        """Post init.

        :rtype: None
        """
        if self.results.return_mode == "path" and self.storage.backend != "sqlite":
            raise ValueError("return_mode='path' requires SQLite storage.")

    def update(
        self,
        *,
        execution: ExecutionSettings | Mapping[str, object] | None = None,
        storage: StorageSettings | Mapping[str, object] | None = None,
        results: ResultsSettings | Mapping[str, object] | None = None,
    ) -> Self:
        """Return validated merged settings without mutating the current run.

        :param execution: Execution settings or a mapping overriding their defaults.
        :type execution: ExecutionSettings | Mapping[str, object] | None
        :param storage: Storage settings or a mapping overriding their defaults.
        :type storage: StorageSettings | Mapping[str, object] | None
        :param results: Result-view settings or a mapping overriding their defaults.
        :type results: ResultsSettings | Mapping[str, object] | None
        :return: Result of this operation.
        :rtype: Self
        """
        changes = {}
        for name, value in (
            ("execution", execution),
            ("storage", storage),
            ("results", results),
        ):
            if value is not None:
                current = getattr(self, name)
                changes[name] = (
                    value
                    if isinstance(value, type(current))
                    else replace(current, **value)
                )
        return replace(self, **changes)

    def to_dict(self) -> dict[str, object]:
        """Return plain configuration values for JSON and future Hydra adapters.

        :return: Result of this operation.
        :rtype: dict[str, object]
        """
        return asdict(self)
