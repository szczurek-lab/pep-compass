"""Regression tests for the separated MUTANG decisions."""

import torch

from pep_compass.autoencoder.geometry import StableTangentGeometry, TangentDecomposition
from pep_compass.autoencoder.subriemannian import PointGeometry
from pep_compass.data.optimization import CandidateBatch, ObjectField
from pep_compass.optimization.components.mutation_generators.strategies.mutang.direction_selection.baseline import BaselineDirectionSelection
from pep_compass.optimization.components.mutation_generators.strategies.mutang.geometry.shared import SharedGeometry
from pep_compass.optimization.components.mutation_generators.strategies.mutang.scoring.max_absolute_loading import MaxAbsoluteLoading


def test_baseline_scoring_matches_directionwise_union() -> None:
    """Maximum absolute loading must represent the historical threshold union."""
    left = torch.tensor([[0.2, 0.1], [0.05, 0.4], [0.3, 0.2], [0.0, 0.1]])
    selected = BaselineDirectionSelection(threshold=0.1, minimum=1).select(
        torch.tensor([1.0, 0.5])
    )

    scores = MaxAbsoluteLoading(max_len=2, alphabet_size=2).score(left, selected)

    assert torch.equal(scores, torch.tensor([[0.2, 0.4], [0.3, 0.1]]))


def test_shared_geometry_rejects_mismatched_point_identity() -> None:
    """Shared MUTANG geometry cannot be attached to a different trajectory point."""
    decomposition = TangentDecomposition(
        torch.eye(2).unsqueeze(0), torch.ones((1, 2)), torch.eye(2).unsqueeze(0)
    )
    geometry = StableTangentGeometry(
        decomposition, 0.01, torch.ones((1, 2), dtype=torch.bool)
    )
    batch = CandidateBatch(
        ["A"],
        torch.zeros((1, 2)),
        {
            "point_id": ObjectField([2]),
            "tangent_geometry": ObjectField([PointGeometry(1, geometry, 7)]),
        },
    )

    try:
        SharedGeometry().resolve(batch)
    except ValueError as error:
        assert "different latent point" in str(error)
    else:
        raise AssertionError("Mismatched point geometry was accepted.")
