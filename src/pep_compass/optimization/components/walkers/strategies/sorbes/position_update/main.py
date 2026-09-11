"""Executable position update preserved from the repository main branch."""

import torch

from pep_compass.optimization.components.walkers.strategies.sorbes.position_update.base import PositionUpdateStrategy


class MainPositionUpdate(PositionUpdateStrategy):
    """Apply ``epsilon*v - 0.5*epsilon**2*a`` with a norm bound."""

    acceleration_coefficient = 0.5

    def __init__(self, *, epsilon: float, delta_max: float) -> None:
        self.epsilon = epsilon
        self.delta_max = delta_max

    def __call__(self, positions, directions, acceleration):
        epsilon = self._bounded_epsilon(directions.active, acceleration)
        update = (
            epsilon[:, None] * directions.active
            - self.acceleration_coefficient * epsilon[:, None].square() * acceleration
            + epsilon[:, None] * directions.inactive
        )  # (B, D)
        return positions + update, epsilon.square()

    def _bounded_epsilon(self, velocity, acceleration):
        nominal = torch.full(
            (velocity.shape[0],), self.epsilon, device=velocity.device, dtype=velocity.dtype
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
