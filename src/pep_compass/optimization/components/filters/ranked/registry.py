"""Registries for reusable ranked-filter scoring and selection stages."""

from pep_compass.optimization.components.filters.ranked.scoring.base import ScoreFunction
from pep_compass.optimization.components.filters.ranked.selection.base import SelectionRule
from pep_compass.registry import Registry


scoring_registry: Registry[ScoreFunction] = Registry(
    "ranked-filter scoring", expected_type=ScoreFunction
)
selection_registry: Registry[SelectionRule] = Registry(
    "ranked-filter selection", expected_type=SelectionRule
)
