"""Controlled configuration reduction for model-backed pipeline test runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True, slots=True)
class TestRunPolicy:
    """Bound expensive operations while preserving pipeline connectivity.

    :param tasks: Maximum number of materialized input tasks to execute.
    :param iterations: Maximum iterations of every loop or local enumeration.
    :param candidates: Maximum generated candidates for one mutation parent.
    :param oracle_calls: Global oracle-call limit.
    """

    tasks: int = 2
    iterations: int = 1
    candidates: int = 2
    oracle_calls: int = 2

    def __post_init__(self) -> None:
        for name in ("tasks", "iterations", "candidates", "oracle_calls"):
            if getattr(self, name) < 1:
                raise ValueError(f"Test-run {name} must be positive.")


def constrain_pipeline_for_test_run(
    pipeline: Mapping[str, Any],
    policy: TestRunPolicy,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return a bounded pipeline mapping and explicit applied overrides."""
    constrained = deepcopy(dict(pipeline))
    overrides: list[str] = []
    limits = constrained.setdefault("limits", {})
    _lower(limits, "oracle_calls", policy.oracle_calls, "pipeline.limits", overrides)
    # REMARK: ``generated_candidates`` is a stop condition, not a per-parent
    # cap. Lowering it to the sample size would stop Local Enumeration after
    # its initial MUTANG call and would not validate the complete pipeline.
    _constrain_steps(
        constrained.get("steps", []),
        policy,
        "pipeline.steps",
        overrides,
    )
    return constrained, tuple(overrides)


def _constrain_steps(
    steps: Any,
    policy: TestRunPolicy,
    path: str,
    overrides: list[str],
) -> None:
    """Recursively constrain known expensive operations in place."""
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping) or len(raw_step) != 1:
            continue
        operation, raw_settings = next(iter(raw_step.items()))
        if not isinstance(raw_settings, dict):
            continue
        step_path = f"{path}.{index}.{operation}"
        if operation == "loop":
            _lower(raw_settings, "iterations", policy.iterations, step_path, overrides)
            _constrain_steps(raw_settings.get("steps", []), policy, step_path, overrides)
        elif operation == "parallel":
            _constrain_steps(raw_settings.get("steps", []), policy, step_path, overrides)
            for branch_index, branch in enumerate(raw_settings.get("branches", [])):
                if isinstance(branch, Mapping):
                    _constrain_steps(
                        branch.get("steps", []),
                        policy,
                        f"{step_path}.branches.{branch_index}",
                        overrides,
                    )
        elif operation == "local_enumeration":
            original_walk_time = raw_settings.pop("walk_time", None)
            if original_walk_time is not None:
                overrides.append(
                    f"{step_path}.walk_time: {original_walk_time!r} -> removed"
                )
            _lower(raw_settings, "iterations", policy.iterations, step_path, overrides)
            _lower(raw_settings, "trajectories", 1, step_path, overrides)
            generator = raw_settings.get("mutation_generator")
            if isinstance(generator, dict):
                parameters = generator.setdefault("parameters", {})
                _lower(
                    parameters,
                    "maximum_candidates",
                    policy.candidates,
                    f"{step_path}.mutation_generator.parameters",
                    overrides,
                )
        elif operation == "mutation_generator":
            parameters = raw_settings.setdefault("parameters", {})
            _lower(
                parameters,
                "maximum_candidates",
                policy.candidates,
                f"{step_path}.parameters",
                overrides,
            )


def _lower(
    mapping: dict[str, Any],
    key: str,
    limit: int,
    path: str,
    overrides: list[str],
) -> None:
    """Set a missing or larger integer limit and record the transformation."""
    original = mapping.get(key)
    if isinstance(original, int) and not isinstance(original, bool) and original <= limit:
        return
    mapping[key] = limit
    override = f"{path}.{key}: {original!r} -> {limit}"
    overrides.append(override)
    logger.info("Test-run override %s.", override)
