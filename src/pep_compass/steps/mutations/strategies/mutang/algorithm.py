"""Fixed internal MUTANG action sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.data_schemas.candidates import CandidateBatch
    from pep_compass.optimization.engine.execution.context import OptimizationContext
    from pep_compass.steps.mutations.strategies.mutang.base import (
        CombinationStrategy,
        DirectionStrategy,
        GeometryStrategy,
        MutationSelectionStrategy,
        ScoringStrategy,
    )
    from pep_compass.steps.mutations.strategies.mutang.schema import MutationOption


class MutationAlgorithm:
    """Compose geometry, direction, scoring, selection, and combination stages."""

    def __init__(
        self,
        geometry: GeometryStrategy,
        directions: DirectionStrategy,
        scoring: ScoringStrategy,
        selection: MutationSelectionStrategy,
        combination: CombinationStrategy,
        autoencoder: Autoencoder | None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param geometry: Geometry strategy or aligned numerical geometry, according to the declared type.
        :type geometry: GeometryStrategy
        :param directions: Direction-selection strategy or sampled numerical directions.
        :type directions: DirectionStrategy
        :param scoring: Strategy computing mutation or candidate scores.
        :type scoring: ScoringStrategy
        :param selection: Strategy selecting admissible substitutions or candidate rows.
        :type selection: MutationSelectionStrategy
        :param combination: Strategy enumerating selected mutation combinations.
        :type combination: CombinationStrategy
        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder | None
        :rtype: None
        """
        self.geometry, self.directions = geometry, directions
        self.scoring, self.selection = scoring, selection
        self.combination, self.autoencoder = combination, autoencoder

    def generate(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> list[tuple[tuple[str, ...], tuple[MutationOption, ...]]]:
        """Generate.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: list[tuple[tuple[str, ...], tuple[MutationOption, ...]]]
        """
        with context.measure_stage(
            "geometry", input_count=len(batch), output_count=len(batch)
        ):
            geometries = self.geometry.resolve(batch, self.autoencoder)
        results = []
        for index, sequence in enumerate(batch.sequences):
            geometry = geometries[index]
            with context.measure_stage("directions", input_count=1):
                selected = self.directions.select(geometry.singular_values[0])
            with context.measure_stage("scoring", input_count=1):
                scores = self.scoring.score(
                    geometry.left_vectors[0], selected
                )  # (L, A)
            with context.measure_stage("selection", input_count=1):
                options = self.selection.select(scores, len(sequence))
            with context.measure_stage("combination", input_count=1):
                generated = tuple(
                    dict.fromkeys(
                        self.combination.generate(sequence, options, context.rng)
                    )
                )
            results.append((generated, options))
        return results
