"""Direct Levenshtein-distance constraint."""

import torch

from pep_compass.data.optimization import ObjectField
from pep_compass.optimization.components.filters.direct.base import DirectFilter
from pep_compass.optimization.components.filters.manager import FilterManager


def levenshtein_distance(left: str, right: str) -> int:
    """Return the exact edit distance using one dynamic-programming row."""
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


@FilterManager.register("levenshtein")
class LevenshteinConstraint(DirectFilter):
    """Keep sequences within an inclusive edit radius of a reference."""

    def __init__(self, reference: str | None = None, maximum_distance: int = 4,
                 reference_field: str | None = None) -> None:
        if maximum_distance < 0:
            raise ValueError("Maximum Levenshtein distance cannot be negative.")
        if (reference is None) == (reference_field is None):
            raise ValueError("Provide exactly one Levenshtein reference or reference_field.")
        self.reference, self.reference_field = reference, reference_field
        self.maximum_distance = maximum_distance

    def _execute(self, batch, context):
        del context
        references = None
        if self.reference_field is not None:
            field = batch.fields.get(self.reference_field)
            if not isinstance(field, ObjectField):
                raise ValueError("Levenshtein reference_field must name an ObjectField.")
            references = field.values
        retained = []
        for index, sequence in enumerate(batch.sequences):
            reference = self.reference if references is None else references[index]
            if levenshtein_distance(sequence, reference) <= self.maximum_distance:
                retained.append(index)
        return batch.select(torch.tensor(retained, device=batch.latent_origins.device))
