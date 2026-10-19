"""Activity model turning a predicted MIC into a quantity that can be integrated.

The proteolysis environment predicts what a peptide is reduced to over time; this
subpackage decides what that reduction means for antimicrobial activity. The two
concerns are kept apart because the step from MIC to activity is a modelling
choice with biological content, and the calibration specification requires it to
be stated explicitly and validated on its own.

Transforms are ordinary callable factories selected explicitly by the caller.
"""

from pep_compass.steps.helpers.activity.transforms import (
    DEFAULT_GROWTH_RATE,
    DEFAULT_HILL_COEFFICIENT,
    DEFAULT_MAX_KILL_RATE,
    ActivityTransform,
    activity_budget,
    hill_kill_rate,
    log_potency,
    net_kill_rate,
    potency_sum,
    potency_sum_transform,
    soft_threshold,
    threshold,
)

__all__ = [
    "DEFAULT_GROWTH_RATE",
    "DEFAULT_HILL_COEFFICIENT",
    "DEFAULT_MAX_KILL_RATE",
    "ActivityTransform",
    "activity_budget",
    "hill_kill_rate",
    "log_potency",
    "net_kill_rate",
    "potency_sum",
    "potency_sum_transform",
    "soft_threshold",
    "threshold",
]
