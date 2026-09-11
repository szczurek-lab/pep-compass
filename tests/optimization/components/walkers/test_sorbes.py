"""Tests for batched SORBES integration strategies."""

from __future__ import annotations

import torch

from pep_compass.optimization.components.walkers.strategies.sorbes.directions.active_inactive import ActiveInactiveDirections
from pep_compass.optimization.components.walkers.strategies.sorbes.geometry.kappa_stable import KappaStableGeometry
from pep_compass.optimization.components.walkers.strategies.sorbes.position_update.article import ArticlePositionUpdate
from pep_compass.optimization.components.walkers.strategies.sorbes.position_update.main import MainPositionUpdate
from pep_compass.optimization.components.walkers.strategies.sorbes.scaling.stable_dimension import StableDimensionScaling


class QuadraticAutoencoder:
    """Two-dimensional decoder with a deterministic directional derivative."""

    def decoder_jacobian(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.eye(2, dtype=positions.dtype).expand(positions.shape[0], -1, -1)

    def field_derivative(
        self, positions: torch.Tensor, directions: torch.Tensor
    ) -> torch.Tensor:
        del positions
        return directions.square()  # (B, A=2)


def test_main_and_article_integrators_preserve_documented_divergence(
    monkeypatch,
) -> None:
    """Article integration must use twice the historical acceleration term."""
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.ones_like(tensor))
    positions = torch.zeros((1, 2), dtype=torch.float32)  # (B=1, D=2)
    geometry = KappaStableGeometry(QuadraticAutoencoder(), kappa=0.01)(positions)
    directions = StableDimensionScaling()(
        ActiveInactiveDirections(vertical_movement=False)(geometry), geometry
    )
    acceleration = directions.active.square()
    main, time_steps = MainPositionUpdate(epsilon=0.1, delta_max=10.0)(
        positions, directions, acceleration
    )
    article, _ = ArticlePositionUpdate(epsilon=0.1, delta_max=10.0)(
        positions, directions, acceleration
    )

    assert main.shape == (1, 2)
    assert torch.all(torch.isfinite(main))
    assert torch.allclose(time_steps, torch.tensor([0.01]))
    assert torch.all(article < main)


def test_zero_active_dimension_keeps_position_finite(monkeypatch) -> None:
    """An empty stable space must stop the row instead of producing NaNs."""
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.ones_like(tensor))
    positions = torch.tensor([[1.0, 2.0]], dtype=torch.float32)  # (B=1, D=2)
    geometry = KappaStableGeometry(QuadraticAutoencoder(), kappa=4.0)(positions)

    assert not geometry.active_mask.any()
