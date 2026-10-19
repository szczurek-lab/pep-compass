"""Optimization-engine adapter for MUTANG."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.data_schemas.candidates import ObjectField, TensorField
from pep_compass.steps.base import Step

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.data_schemas.candidates import CandidateBatch
    from pep_compass.data_schemas.contracts import StepContract
    from pep_compass.optimization.engine.execution.context import OptimizationContext
    from pep_compass.steps.mutations.strategies.mutang.base import (
        CombinationStrategy,
        DirectionStrategy,
        GeometryStrategy,
        MutationSelectionStrategy,
        ScoringStrategy,
    )


class Mutang(Step):
    """Convert the internal MUTANG results into an aligned candidate batch."""

    stage_parameters = ("geometry", "directions", "scoring", "selection", "combination")
    category = "mutation_generator"

    required_resources = ("autoencoder",)
    output_fields = (
        "mutation.parent_sequence",
        "mutation.options",
        "mutation.scored_options",
    )
    invalidated_prefixes = ("oracle.",)
    display_parameters = ("maximum_candidates", "geometry")

    def __init__(
        self,
        *,
        maximum_candidates: int | None = None,
        geometry: GeometryStrategy | None = None,
        directions: DirectionStrategy | None = None,
        scoring: ScoringStrategy | None = None,
        selection: MutationSelectionStrategy | None = None,
        combination: CombinationStrategy | None = None,
        max_len: int = 25,
        direction_significance_threshold: float = 0.001,
        min_number_of_directions: int = 5,
        token_threshold: float = 0.1,
        alphabet: Sequence[str] | None = None,
    ) -> None:
        """Configure independent mutation stages without model initialization.

        :param maximum_candidates: Per-parent allocation cap, or None for unbounded generation.
        :param geometry: Shared local geometry by default; pass an independent geometry explicitly.
        :param combination: Explicit combination strategy; its cap must agree with the outer cap.

        :type maximum_candidates: int | None
        :type geometry: GeometryStrategy | None
        :param directions: Direction-selection strategy or sampled numerical directions.
        :type directions: DirectionStrategy | None
        :param scoring: Strategy computing mutation or candidate scores.
        :type scoring: ScoringStrategy | None
        :param selection: Strategy selecting admissible substitutions or candidate rows.
        :type selection: MutationSelectionStrategy | None
        :type combination: CombinationStrategy | None
        :param max_len: Max len.
        :type max_len: int
        :param direction_significance_threshold: Direction significance threshold.
        :type direction_significance_threshold: float
        :param min_number_of_directions: Min number of directions.
        :type min_number_of_directions: int
        :param token_threshold: Token threshold.
        :type token_threshold: float
        :param alphabet: Ordered residue alphabet.
        :type alphabet: Sequence[str] | None
        :rtype: None
        """
        from pep_compass.utils.validation import validate_integer

        from .combination.baseline import BaselineCombination
        from .direction_selection.baseline import BaselineDirectionSelection
        from .geometry.shared import SharedGeometry
        from .mutation_selection.threshold import ThresholdSelection
        from .scoring.max_absolute_loading import MaxAbsoluteLoading

        validate_integer(
            maximum_candidates, "Mutang.maximum_candidates", 1, optional=True
        )
        alphabet = (
            list(alphabet) if alphabet is not None else list(" ACDEFGHIKLMNPQRSTVWY")
        )
        self.geometry = geometry if geometry is not None else SharedGeometry()
        self.directions = (
            directions
            if directions is not None
            else BaselineDirectionSelection(
                direction_significance_threshold, min_number_of_directions
            )
        )
        self.scoring = (
            scoring
            if scoring is not None
            else MaxAbsoluteLoading(max_len, len(alphabet))
        )
        self.selection = (
            selection if selection is not None else ThresholdSelection(token_threshold)
        )
        self.combination = (
            combination
            if combination is not None
            else BaselineCombination(alphabet, maximum_candidates)
        )
        cap = getattr(self.combination, "maximum_candidates", None)
        if maximum_candidates is not None and cap != maximum_candidates:
            raise ValueError(
                "Mutang maximum_candidates conflicts with its combination cap."
            )
        self.maximum_candidates = cap
        self._configure(
            dict(
                maximum_candidates=cap,
                geometry=self.geometry,
                directions=self.directions,
                scoring=self.scoring,
                selection=self.selection,
                combination=self.combination,
                max_len=max_len,
                direction_significance_threshold=direction_significance_threshold,
                min_number_of_directions=min_number_of_directions,
                token_threshold=token_threshold,
                alphabet=alphabet,
            )
        )
        self.geometry_requirement = getattr(self.geometry, "requirement", None)
        self.requires_local_enumeration = self.geometry.local_enumeration_only

    def contract(self) -> StepContract:
        """Contract.

        :return: Result of this operation.
        :rtype: StepContract
        """
        from pep_compass.data_schemas.contracts import StepContract

        return StepContract(
            ("tangent_geometry", "point_id") if self.requires_local_enumeration else (),
            produced_fields=self.output_fields,
            invalidated_prefixes=self.invalidated_prefixes,
            resources=self.required_resources,
        )

    def candidate_bound(self, count: int) -> object:
        """Candidate bound.

        :param count: Count.
        :type count: int
        :return: Result of this operation.
        :rtype: object
        """
        return (
            None if self.maximum_candidates is None else count * self.maximum_candidates
        )

    def _initialize(self, autoencoder: Autoencoder | None) -> None:
        """Initialize.

        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder | None
        :rtype: None
        """
        from .algorithm import MutationAlgorithm

        self.mutang = MutationAlgorithm(
            self.geometry,
            self.directions,
            self.scoring,
            self.selection,
            self.combination,
            autoencoder,
        )

    def _execute(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> CandidateBatch:
        """Execute.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        limit = context.state.limits.generated_candidates
        if limit is not None and context.state.generated_candidates >= limit:
            context.state.stop_requested = True
            return batch
        sequences, parents, parent_indices, options, scored_options = [], [], [], [], []
        for index, (generated, selected) in enumerate(
            self.mutang.generate(batch, context)
        ):
            sequences.extend(generated)
            parents.extend([batch.sequences[index]] * len(generated))
            parent_indices.extend([index] * len(generated))
            grouped = {}
            for option in selected:
                grouped.setdefault(option.position, []).append(option.residue)
            options.extend([grouped] * len(generated))
            scored_options.extend([selected] * len(generated))
        indices = torch.as_tensor(parent_indices, device=batch.latent_origins.device)
        result = batch.repeat_from_parents(indices).with_sequences(sequences)
        parent_ids = batch.fields.get("lineage.candidate_id")
        if isinstance(parent_ids, TensorField):
            result = result.with_field(
                "lineage.parent_candidate_id",
                TensorField(parent_ids.values.index_select(0, indices)),
            )
        result = result.with_field(
            "lineage.candidate_id",
            TensorField(
                torch.as_tensor(
                    context.state.next_candidate_ids(len(result)),
                    dtype=torch.long,
                    device=result.latent_origins.device,
                )
            ),
        )
        context.state.record_generated_candidates(len(result))
        result = result.with_field("mutation.parent_sequence", ObjectField(parents))
        result = result.with_field("mutation.options", ObjectField(options))
        return result.with_field("mutation.scored_options", ObjectField(scored_options))
