"""Tests for ranked and direct filter contracts."""

import torch

from pep_compass.data.optimization import CandidateBatch, ObjectField
from pep_compass.optimization.components.filters.direct.constraints.levenshtein import LevenshteinConstraint
from pep_compass.optimization.components.filters.direct.constraints.sequence_length import SequenceLengthConstraint
from pep_compass.optimization.components.filters.ranked.base import RankedFilter
from pep_compass.optimization.components.filters.ranked.scoring.base import ScoreFunction
from pep_compass.optimization.components.filters.ranked.selection.nucleus import NucleusSelection
from pep_compass.optimization.components.filters.ranked.selection.threshold import ThresholdSelection
from pep_compass.optimization.engine.execution.context import OptimizationContext


class FixedScores(ScoreFunction):
    """Return deterministic test scores."""

    def __call__(self, batch, context):
        del context
        return torch.tensor([0.1, 0.8, 0.4], device=batch.latent_origins.device)


def _batch():
    return CandidateBatch(["AA", "AAA", "AABA"], torch.zeros((3, 2)))


def test_ranked_filter_composes_scoring_and_threshold_selection() -> None:
    result = RankedFilter(FixedScores(), ThresholdSelection(0.4))(
        _batch(), OptimizationContext(autoencoder=None)
    )
    assert result.sequences == ("AAA", "AABA")


def test_nucleus_selection_keeps_best_candidate_first() -> None:
    indices = NucleusSelection(top_p=0.5).select(
        torch.tensor([0.0, 4.0, 1.0]), higher_is_better=True, context=None
    )
    assert indices.tolist() == [1]


def test_nucleus_excludes_explicitly_unscored_rows_at_full_mass() -> None:
    indices = NucleusSelection(top_p=1.0).select(
        torch.tensor([0.0, float("-inf"), 1.0]),
        higher_is_better=True,
        context=None,
    )
    assert indices.tolist() == [2, 0]


def test_direct_sequence_constraints_preserve_aligned_rows() -> None:
    context = OptimizationContext(autoencoder=None)
    by_length = SequenceLengthConstraint(minimum=3, maximum=3)(_batch(), context)
    by_distance = LevenshteinConstraint("AAA", maximum_distance=1)(_batch(), context)
    assert by_length.sequences == ("AAA",)
    assert by_distance.sequences == ("AA", "AAA", "AABA")


def test_levenshtein_constraint_uses_per_trajectory_center_field() -> None:
    batch = CandidateBatch(
        ["AAAA", "CCCC"],
        torch.zeros((2, 2)),
        {"center": ObjectField(["AAAT", "GGGG"])},
    )

    result = LevenshteinConstraint(
        reference_field="center", maximum_distance=1
    )(batch, OptimizationContext(autoencoder=None))

    assert result.sequences == ("AAAA",)
