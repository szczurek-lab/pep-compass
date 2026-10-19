"""Executable position update preserved from the repository main branch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.steps.walkers.strategies.sorbes.position_update.base import (
    PositionUpdateStrategy,
)

if TYPE_CHECKING:
    from pep_compass.steps.walkers.strategies.sorbes.schema import SorbesDirections


class MainPositionUpdate(PositionUpdateStrategy):
    """Apply ``epsilon*v - 0.5*epsilon**2*a`` with a norm bound."""

    acceleration_coefficient = 0.5

    def __init__(self, *, epsilon: float, delta_max: float) -> None:
        """Initialize this object with the declared configuration.

        :param epsilon: Epsilon.
        :type epsilon: float
        :param delta_max: Delta max.
        :type delta_max: float
        :rtype: None
        """
        from pep_compass.utils.validation import validate_positive

        validate_positive(epsilon, "MainPositionUpdate.epsilon")
        validate_positive(delta_max, "MainPositionUpdate.delta_max")
        self._configure({"epsilon": epsilon, "delta_max": delta_max})
        self.epsilon = epsilon
        self.delta_max = delta_max

    def __call__(
        self,
        positions: torch.Tensor,
        directions: SorbesDirections,
        acceleration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Call.

        :param positions: Positions.
        :type positions: torch.Tensor
        :param directions: Direction-selection strategy or sampled numerical directions.
        :type directions: SorbesDirections
        :param acceleration: Acceleration.
        :type acceleration: torch.Tensor
        :return: Result of this operation.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        epsilon = self._bounded_epsilon(directions.active, acceleration)
        update = (
            epsilon[:, None] * directions.active
            - self.acceleration_coefficient * epsilon[:, None].square() * acceleration
            + epsilon[:, None] * directions.inactive
        )  # (B, D)
        return positions + update, epsilon.square()

    def _bounded_epsilon(
        self, velocity: torch.Tensor, acceleration: torch.Tensor
    ) -> torch.Tensor:
        """Bounded epsilon.

        :param velocity: Velocity.
        :type velocity: torch.Tensor
        :param acceleration: Acceleration.
        :type acceleration: torch.Tensor
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        nominal = torch.full(
            (velocity.shape[0],),
            self.epsilon,
            device=velocity.device,
            dtype=velocity.dtype,
        )
        low, high = torch.zeros_like(nominal), nominal.clone()
        for _ in range(32):
            value = (low + high) * 0.5
            update = value[:, None] * velocity - (
                self.acceleration_coefficient * value[:, None].square() * acceleration
            )
            fits = torch.linalg.vector_norm(update, dim=1) <= self.delta_max
            low, high = torch.where(fits, value, low), torch.where(fits, high, value)
        full_update = nominal[:, None] * velocity - (
            self.acceleration_coefficient * nominal[:, None].square() * acceleration
        )
        return torch.where(
            torch.linalg.vector_norm(full_update, dim=1) <= self.delta_max,
            nominal,
            low,
        )
