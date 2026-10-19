"""Direct Levenshtein-distance constraint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.data_schemas.candidates import ObjectField
from pep_compass.steps.base import Step

if TYPE_CHECKING:
    from pep_compass.data_schemas.candidates import CandidateBatch
    from pep_compass.optimization.engine.execution.context import OptimizationContext


def levenshtein_distance(left: str, right: str) -> int:
    """Return the exact edit distance using one dynamic-programming row.

    :param left: Left.
    :type left: str
    :param right: Right.
    :type right: str
    :return: Result of this operation.
    :rtype: int
    """
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


class Levenshtein(Step):
    """Keep sequences within an inclusive edit radius of a reference."""

    category = "filter"
    input_fields = staticmethod(
        lambda values: (
            (values["reference_field"],) if values.get("reference_field") else ()
        )
    )
    display_parameters = ("maximum_distance", "reference", "reference_field")

    def __init__(
        self,
        reference: str | None = None,
        maximum_distance: int = 4,
        reference_field: str | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param reference: Reference.
        :type reference: str | None
        :param maximum_distance: Maximum distance.
        :type maximum_distance: int
        :param reference_field: Reference field.
        :type reference_field: str | None
        :rtype: None
        """
        self._configure(
            {
                "reference": reference,
                "maximum_distance": maximum_distance,
                "reference_field": reference_field,
            }
        )
        if maximum_distance < 0:
            raise ValueError("Maximum Levenshtein distance cannot be negative.")
        if (reference is None) == (reference_field is None):
            raise ValueError(
                "Provide exactly one Levenshtein reference or reference_field."
            )
        self.reference, self.reference_field = reference, reference_field
        self.maximum_distance = maximum_distance

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
        del context
        references = None
        if self.reference_field is not None:
            field = batch.fields.get(self.reference_field)
            if not isinstance(field, ObjectField):
                raise ValueError(
                    "Levenshtein reference_field must name an ObjectField."
                )
            references = field.values
        retained = []
        for index, sequence in enumerate(batch.sequences):
            reference = self.reference if references is None else references[index]
            if levenshtein_distance(sequence, reference) <= self.maximum_distance:
                retained.append(index)
        return batch.select(torch.tensor(retained, device=batch.latent_origins.device))
