"""Typed values exchanged by the fixed SORBES stages."""

from dataclasses import dataclass

import torch

from pep_compass.autoencoder.geometry import StableTangentGeometry


@dataclass(frozen=True, slots=True)
class SorbesDirections:
    """Active and inactive latent directions, each shaped ``(B, D)``."""

    active: torch.Tensor
    inactive: torch.Tensor


@dataclass(frozen=True, slots=True)
class SorbesStepResult:
    """New point and geometry belonging to that new point."""

    positions: torch.Tensor
    adjusted_time_steps: torch.Tensor
    geometry: StableTangentGeometry
