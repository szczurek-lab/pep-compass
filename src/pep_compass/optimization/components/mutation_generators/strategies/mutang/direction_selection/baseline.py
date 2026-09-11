"""Historical MUTANG tangent-direction selection."""

import torch

from pep_compass.optimization.components.mutation_generators.strategies.mutang.schema import SelectedDirections


class BaselineDirectionSelection:
    """Preserve the main/dev significance threshold and minimum count."""

    def __init__(self, threshold=1e-3, minimum=5):
        self.threshold, self.minimum = threshold, minimum

    def select(self, singular_values):
        significant = int((singular_values > self.threshold).sum().item())
        count = min(max(significant, self.minimum), singular_values.numel())
        return SelectedDirections(torch.arange(count, device=singular_values.device))
