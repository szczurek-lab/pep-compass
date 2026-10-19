"""Base contracts for filters driven by reusable candidate scores."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pep_compass.steps.base import Step
from pep_compass.steps.filters.strategies.ranked.scoring.base import ScoreFunction
from pep_compass.steps.filters.strategies.ranked.selection.base import SelectionRule

if TYPE_CHECKING:
    from pep_compass.data_schemas.candidates import CandidateBatch
    from pep_compass.data_schemas.contracts import StepContract
    from pep_compass.optimization.engine.execution.context import OptimizationContext


class Ranked(Step):
    """Compose one score function with one score-based selection rule."""

    stage_parameters = ("scoring", "selection")
    category = "filter"

    def __init__(self, scoring: ScoreFunction, selection: SelectionRule) -> None:
        """Initialize this object with the declared configuration.

        :param scoring: Strategy computing mutation or candidate scores.
        :type scoring: ScoreFunction
        :param selection: Strategy selecting admissible substitutions or candidate rows.
        :type selection: SelectionRule
        :rtype: None
        """
        self._configure({"scoring": scoring, "selection": selection})
        self.scoring, self.selection = scoring, selection

    required_resources = ("autoencoder",)

    def contract(self) -> StepContract:
        """Contract.

        :return: Result of this operation.
        :rtype: StepContract
        """
        from pep_compass.data_schemas.contracts import StepContract

        contract = getattr(self.scoring, "contract", None)
        return (
            contract()
            if contract is not None
            else StepContract(resources=self.required_resources)
        )

    def _execute(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> CandidateBatch:
        """Score candidates and apply the configured selection strategy.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: CandidateBatch
        """
        with context.measure_stage(
            "scoring", input_count=len(batch), output_count=len(batch)
        ):
            scores = self.scoring(batch, context)  # (B,)
        if scores.ndim != 1 or scores.shape[0] != len(batch):
            raise ValueError(
                "Ranked filter scoring must return one score per candidate."
            )
        with context.measure_stage("selection", input_count=len(batch)):
            indices = self.selection.select(
                scores, higher_is_better=self.scoring.higher_is_better, context=context
            )
        return batch.select(indices)
