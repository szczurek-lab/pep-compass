"""Temperature-scaled nucleus selection."""

import torch

from pep_compass.optimization.components.filters.ranked.selection.base import SelectionRule


class NucleusSelection(SelectionRule):
    """Keep the smallest best-first prefix reaching configured probability mass."""

    def __init__(self, top_p=0.9, temperature=1.0) -> None:
        if not 0 < top_p <= 1 or temperature <= 0:
            raise ValueError("Nucleus top_p and temperature must be positive.")
        self.top_p, self.temperature = top_p, temperature

    def select(self, scores, *, higher_is_better, context):
        del context
        if scores.numel() == 0:
            return torch.arange(scores.numel(), device=scores.device)
        oriented = scores if higher_is_better else -scores
        order = torch.argsort(oriented, descending=True)
        ordered = oriented.index_select(0, order)
        if self.top_p == 1:
            return order[~torch.isneginf(ordered)]
        positive_inf = torch.isposinf(ordered)
        if positive_inf.any():
            probabilities = positive_inf.to(scores.dtype) / positive_inf.sum()
        elif not torch.isfinite(ordered).any():
            probabilities = torch.full_like(ordered, 1.0 / ordered.numel())
        else:
            finite = torch.isfinite(ordered)
            probabilities = torch.zeros_like(ordered)
            probabilities[finite] = torch.softmax(
                ordered[finite] / self.temperature, dim=0
            )
        cumulative = torch.cumsum(probabilities, dim=0)
        keep = torch.ones_like(cumulative, dtype=torch.bool)
        keep[1:] = cumulative[:-1] < self.top_p
        return order[keep]
