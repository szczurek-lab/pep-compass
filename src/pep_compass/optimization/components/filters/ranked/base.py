"""Base contracts for filters driven by reusable candidate scores."""

from pep_compass.optimization.components.filters.base import Filter
from pep_compass.optimization.components.filters.ranked.scoring.base import ScoreFunction
from pep_compass.optimization.components.filters.ranked.selection.base import SelectionRule


class RankedFilter(Filter):
    """Compose one score function with one score-based selection rule."""

    def __init__(self, scoring: ScoreFunction, selection: SelectionRule) -> None:
        self.scoring, self.selection = scoring, selection

    def _execute(self, batch, context):
        scores = self.scoring(batch, context)  # (B,)
        if scores.ndim != 1 or scores.shape[0] != len(batch):
            raise ValueError("Ranked filter scoring must return one score per candidate.")
        indices = self.selection.select(
            scores, higher_is_better=self.scoring.higher_is_better, context=context
        )
        return batch.select(indices)
