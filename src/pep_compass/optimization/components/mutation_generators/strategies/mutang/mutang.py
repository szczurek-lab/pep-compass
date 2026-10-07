"""Fixed internal MUTANG action sequence."""

from pep_compass.autoencoder.base import Autoencoder
from pep_compass.data.optimization import CandidateBatch
from pep_compass.optimization.components.mutation_generators.strategies.mutang.combination.baseline import (
    BaselineCombination,
)
from pep_compass.optimization.components.mutation_generators.strategies.mutang.direction_selection.baseline import (
    BaselineDirectionSelection,
)
from pep_compass.optimization.components.mutation_generators.strategies.mutang.geometry.kappa_stable import (
    KappaStableGeometry,
)
from pep_compass.optimization.components.mutation_generators.strategies.mutang.mutation_selection.threshold import (
    ThresholdSelection,
)
from pep_compass.optimization.components.mutation_generators.strategies.mutang.scoring.max_absolute_loading import (
    MaxAbsoluteLoading,
)
from pep_compass.optimization.engine.execution.context import OptimizationContext


class Mutang:
    """Compose geometry, direction, scoring, selection, and combination stages."""

    def __init__(
        self,
        geometry: KappaStableGeometry,
        directions: BaselineDirectionSelection,
        scoring: MaxAbsoluteLoading,
        selection: ThresholdSelection,
        combination: BaselineCombination,
        autoencoder: Autoencoder,
    ):
        self.geometry, self.directions = geometry, directions
        self.scoring, self.selection = scoring, selection
        self.combination, self.autoencoder = combination, autoencoder

    def generate(self, batch: CandidateBatch, context: OptimizationContext):
        geometries = self.geometry.resolve(batch, self.autoencoder)
        results = []
        for index, sequence in enumerate(batch.sequences):
            geometry = geometries[index]
            selected = self.directions.select(geometry.singular_values[0])
            scores = self.scoring.score(geometry.left_vectors[0], selected)
            options = self.selection.select(scores, len(sequence))
            generated = tuple(
                dict.fromkeys(self.combination.generate(sequence, options, context.rng))
            )
            results.append((generated, options))
        return results
