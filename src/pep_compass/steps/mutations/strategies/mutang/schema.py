"""Typed intermediate values of the MUTANG algorithm."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MutationOption:
    """One scored residue substitution."""

    position: int
    residue: int
    score: float


@dataclass(frozen=True, slots=True)
class SelectedDirections:
    """Indices selected from tangent directions."""

    indices: torch.Tensor
