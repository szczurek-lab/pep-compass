"""Materialize pipeline parameter-grid variants."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True, slots=True)
class PipelineVariant:
    """Contain one concrete pipeline mapping generated from a grid."""

    index: int
    values: Mapping[str, Any]
    pipeline: Mapping[str, Any]

    @property
    def variant_id(self) -> str:
        """Return a stable filesystem-safe variant identifier."""
        return f"variant_{self.index:05d}"


def materialize_variants(
    pipeline: Mapping[str, Any],
    grid: Mapping[str, tuple[Any, ...]],
) -> tuple[PipelineVariant, ...]:
    """Apply a Cartesian parameter grid to a pipeline mapping."""
    if not grid:
        return (PipelineVariant(0, {}, deepcopy(pipeline)),)
    paths = tuple(grid)
    if any(not values for values in grid.values()):
        raise ValueError("Grid value sequences cannot be empty.")
    variants = []
    for index, values in enumerate(product(*(grid[path] for path in paths))):
        selected = dict(zip(paths, values, strict=True))
        concrete = deepcopy(pipeline)
        for path, value in selected.items():
            _set_path(concrete, path, value)
        variants.append(PipelineVariant(index, selected, concrete))
    return tuple(variants)


def _set_path(configuration: dict[str, Any], path: str, value: Any) -> None:
    """Set one existing dotted path in a nested mapping/list structure."""
    parts = path.split(".")
    target: Any = configuration
    for part in parts[:-1]:
        if isinstance(target, dict) and part in target:
            target = target[part]
        elif isinstance(target, list) and part.isdigit() and int(part) < len(target):
            target = target[int(part)]
        else:
            raise ValueError(f"Grid path does not exist: {path}")
    leaf = parts[-1]
    if isinstance(target, dict) and leaf in target:
        target[leaf] = value
    elif isinstance(target, list) and leaf.isdigit() and int(leaf) < len(target):
        target[int(leaf)] = value
    else:
        raise ValueError(f"Grid path does not exist: {path}")
