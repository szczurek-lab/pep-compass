r"""Turning predicted MIC into an activity that can be integrated over time.

## The problem

The time-course model of the calibration needs a budget

.. math::
    B(T) = \int_0^T A(t)\,\mathrm{d}t,

where :math:`A(t)` is the antimicrobial activity of whatever the peptide has been
reduced to at time :math:`t`. The predicted quantity is a MIC, and a MIC cannot
be integrated: it is inversely related to activity, its units are a
concentration, and the integral of a concentration over time is not a meaningful
quantity.

This module defines the step from MIC to activity explicitly, offers several
constructions of it, and keeps them behind one registry so that an analysis can
be repeated under each without changing anything else.

## The state variable

A peptide under proteolysis is not one molecule but a mixture: the surviving
parent and the fragments produced so far. Each species :math:`i` is present at a
molar concentration :math:`C_i` and carries a predicted :math:`\mathrm{MIC}_i`.

The mixture is summarized by the **potency sum**

.. math::
    \psi = \sum_i \frac{C_i}{\mathrm{MIC}_i},

which is the classical fractional inhibitory concentration. It is dimensionless,
it equals one exactly at the inhibitory boundary, and it is additive over
species, which is what makes it usable as the state of a cleavage process: a cut
replaces one term by two, and nothing else in the expression changes.

## Molar bookkeeping

One parent molecule cut once yields one molecule of each fragment. A fragment
therefore inherits the **molar concentration** of the species it came from; the
concentration is not divided between the products. Mass is conserved, molar
concentration per species is not, and it is the molar form that enters
:math:`\psi`.

## Why a saturating kill rate is the default

Reading :math:`A` as a rate of killing is what makes :math:`B(T)` interpretable:
its units become cumulative log-kill, so the budget answers "how much killing
does this peptide achieve before proteolysis removes it". The pharmacodynamic
:math:`E_{\max}` form

.. math::
    A(\psi) = E_{\max} \frac{\psi^{H}}{\psi^{H} + 1}

is the standard description of that rate. It has the property the problem
requires -- a lower MIC raises :math:`\psi` and kills faster -- and it saturates,
because no concentration kills faster than the maximum rate of the mechanism.
Without saturation, an arbitrarily potent fragment would contribute an
arbitrarily large amount to the integral, and the budget would be dominated by
whichever MIC the predictor happened to place lowest.

Dimension symbols: ``S`` species in the mixture.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from pep_compass.registry import Registry

# An activity transform maps the potency sum of a mixture to an activity.
ActivityTransform = Callable[[np.ndarray], np.ndarray]

activity_transform_registry: Registry[ActivityTransform] = Registry("activity transform")

# Typical values, stated once so that every transform shares the same defaults.
# REMARK: The Hill coefficient of one and a maximum kill rate of one are
# deliberately neutral: they make the default transform a plain saturating
# function of the potency sum, and any study-specific value must be passed in.
DEFAULT_HILL_COEFFICIENT = 1.0
DEFAULT_MAX_KILL_RATE = 1.0
# REMARK: Half the maximum kill rate is not an arbitrary default. A MIC is by
# definition the concentration at which growth is just inhibited, so the net rate
# must vanish at the inhibitory boundary. The Hill form reaches half its maximum
# at that boundary, so a growth rate of E_max / 2 places the net-zero point
# exactly at psi = 1 and makes the transform agree with the definition of the
# quantity it consumes.
DEFAULT_GROWTH_RATE = DEFAULT_MAX_KILL_RATE / 2.0


def potency_sum(
    concentrations: np.ndarray | Sequence[float],
    mic_values: np.ndarray | Sequence[float],
) -> np.ndarray:
    r"""Summarize a mixture of species by its fractional inhibitory concentration.

    :param concentrations: Molar concentration of each species, shape ``(..., S)``.
    :param mic_values: Predicted MIC of each species in the same units,
        shape ``(..., S)``.
    :return: Potency sum :math:`\psi`, shape ``(...)``.
    :raises ValueError: If a MIC is not strictly positive, which would make the
        ratio undefined.
    """

    concentration_array = np.asarray(concentrations, dtype=np.float64)  # (..., S)
    mic_array = np.asarray(mic_values, dtype=np.float64)  # (..., S)
    if np.any(mic_array <= 0.0):
        raise ValueError("every MIC must be strictly positive.")
    return (concentration_array / mic_array).sum(axis=-1)  # (...)


def _as_array(potency: np.ndarray | Sequence[float] | float) -> np.ndarray:
    """Coerce a potency sum argument into a float array.

    :param potency: Potency sum, any shape.
    :return: Float array of the same shape.
    :raises ValueError: If the potency sum is negative.
    """

    values = np.asarray(potency, dtype=np.float64)
    if np.any(values < 0.0):
        raise ValueError("the potency sum cannot be negative.")
    return values


@activity_transform_registry.register("hill_kill_rate")
def hill_kill_rate(
    hill_coefficient: float = DEFAULT_HILL_COEFFICIENT,
    max_kill_rate: float = DEFAULT_MAX_KILL_RATE,
) -> ActivityTransform:
    r"""Build the saturating pharmacodynamic kill rate.

    .. math::
        A(\psi) = E_{\max} \frac{\psi^{H}}{\psi^{H} + 1}

    At :math:`\psi = 1`, the inhibitory boundary, the rate is half its maximum
    whatever the Hill coefficient, so the parameter changes the sharpness of the
    transition and not its location.

    :param hill_coefficient: Steepness :math:`H` of the transition. Larger values
        approach a step at the inhibitory boundary.
    :param max_kill_rate: Maximum rate :math:`E_{\max}`, in units of activity per
        unit time. Fixes the unit of the integrated budget.
    :return: Transform mapping a potency sum to a kill rate.
    :raises ValueError: If either parameter is not strictly positive.
    """

    if hill_coefficient <= 0.0 or max_kill_rate <= 0.0:
        raise ValueError("hill_coefficient and max_kill_rate must be strictly positive.")

    def transform(potency: np.ndarray) -> np.ndarray:
        values = _as_array(potency)
        raised = np.power(values, hill_coefficient)
        return max_kill_rate * raised / (raised + 1.0)

    return transform


@activity_transform_registry.register("net_kill_rate")
def net_kill_rate(
    hill_coefficient: float = DEFAULT_HILL_COEFFICIENT,
    max_kill_rate: float = DEFAULT_MAX_KILL_RATE,
    growth_rate: float = DEFAULT_GROWTH_RATE,
) -> ActivityTransform:
    r"""Build the kill rate net of bacterial regrowth.

    .. math::
        A(\psi) = E_{\max} \frac{\psi^{H}}{\psi^{H} + 1} - \mu

    The subtracted growth rate :math:`\mu` makes the activity **negative** once
    proteolysis has reduced the mixture below the potency at which killing
    outpaces division. The budget then stops accumulating and begins to fall,
    which is the behaviour a peptide that has been degraded actually shows.

    The potency at which the two balance,
    :math:`\psi^{*} = (\mu / (E_{\max} - \mu))^{1/H}`, is the point the time
    course has to stay above, and it is a more meaningful failure criterion than
    an arbitrary threshold on MIC. At the default :math:`\mu = E_{\max}/2` that
    point is :math:`\psi^{*} = 1` for any Hill coefficient, which is the
    definition of the MIC itself.

    :param hill_coefficient: Steepness :math:`H` of the transition.
    :param max_kill_rate: Maximum kill rate :math:`E_{\max}`.
    :param growth_rate: Bacterial growth rate :math:`\mu` in the same units.
    :return: Transform mapping a potency sum to a net rate.
    :raises ValueError: If the parameters are not positive, or if the growth rate
        is not below the maximum kill rate, in which case no potency can ever
        produce net killing.
    """

    if growth_rate <= 0.0:
        raise ValueError("growth_rate must be strictly positive.")
    if growth_rate >= max_kill_rate:
        raise ValueError(
            "growth_rate must be below max_kill_rate; otherwise no mixture can kill."
        )
    kill = hill_kill_rate(hill_coefficient, max_kill_rate)

    def transform(potency: np.ndarray) -> np.ndarray:
        return kill(potency) - growth_rate

    return transform


@activity_transform_registry.register("potency_sum")
def potency_sum_transform() -> ActivityTransform:
    r"""Build the identity transform, :math:`A(\psi) = \psi`.

    Activity is taken to be proportional to the summed reciprocal MIC. It is the
    simplest reading and is included as a reference, not as a recommendation: it
    is unbounded, so one fragment with a very low predicted MIC can dominate the
    budget regardless of how briefly it exists.

    :return: Transform mapping a potency sum to itself.
    """

    def transform(potency: np.ndarray) -> np.ndarray:
        return _as_array(potency)

    return transform


@activity_transform_registry.register("log_potency")
def log_potency(floor: float = 1e-6) -> ActivityTransform:
    r"""Build the logarithmic potency, :math:`A(\psi) = \log_2 \psi`.

    MIC is measured on a two-fold dilution series, so a doubling of :math:`\psi`
    is one experimental step whether it happens at high or low potency. This
    transform is the one whose unit matches how the underlying quantity was
    obtained.

    It is negative below the inhibitory boundary and unbounded below, so the
    argument is floored.

    :param floor: Smallest potency sum evaluated, below which the value is
        clamped.
    :return: Transform mapping a potency sum to a base-two logarithm.
    :raises ValueError: If the floor is not strictly positive.
    """

    if floor <= 0.0:
        raise ValueError("floor must be strictly positive.")

    def transform(potency: np.ndarray) -> np.ndarray:
        return np.log2(np.maximum(_as_array(potency), floor))

    return transform


@activity_transform_registry.register("threshold")
def threshold(inhibitory_potency: float = 1.0) -> ActivityTransform:
    r"""Build the indicator of inhibition, :math:`A(\psi) = \mathbb{1}[\psi \ge \psi_0]`.

    The mixture either reaches the inhibitory boundary or it does not. The
    integrated budget becomes the **time spent inhibitory**, in units of time,
    which is the quantity clinical breakpoint reasoning uses.

    Its drawback is that it discards every distinction above and below the
    boundary, so two peptides differing by a factor of a hundred in potency score
    identically.

    :param inhibitory_potency: Potency sum at which inhibition begins.
    :return: Transform mapping a potency sum to zero or one.
    :raises ValueError: If the boundary is not strictly positive.
    """

    if inhibitory_potency <= 0.0:
        raise ValueError("inhibitory_potency must be strictly positive.")

    def transform(potency: np.ndarray) -> np.ndarray:
        return (_as_array(potency) >= inhibitory_potency).astype(np.float64)

    return transform


@activity_transform_registry.register("soft_threshold")
def soft_threshold(
    inhibitory_potency: float = 1.0,
    width_in_doublings: float = 1.0,
) -> ActivityTransform:
    r"""Build a smooth indicator of inhibition, logistic in :math:`\log_2 \psi`.

    .. math::
        A(\psi) = \left[1 + 2^{-(\log_2 \psi - \log_2 \psi_0)/w}\right]^{-1}

    The transition is placed on the dilution scale, so ``width_in_doublings``
    states how many two-fold steps the boundary is blurred over. One step is the
    resolution of a MIC assay, which makes it the natural default: the transform
    then treats differences the experiment could not have resolved as
    indistinguishable.

    :param inhibitory_potency: Potency sum at the midpoint of the transition.
    :param width_in_doublings: Width of the transition in two-fold dilution steps.
    :return: Transform mapping a potency sum into the unit interval.
    :raises ValueError: If either parameter is not strictly positive.
    """

    if inhibitory_potency <= 0.0 or width_in_doublings <= 0.0:
        raise ValueError("inhibitory_potency and width_in_doublings must be positive.")

    def transform(potency: np.ndarray) -> np.ndarray:
        values = np.maximum(_as_array(potency), np.finfo(np.float64).tiny)
        exponent = (np.log2(values) - np.log2(inhibitory_potency)) / width_in_doublings
        return 1.0 / (1.0 + np.power(2.0, -exponent))

    return transform


def build_activity_transform(name: str, **parameters: float) -> ActivityTransform:
    """Construct a registered activity transform by name.

    :param name: Registered transform name.
    :param parameters: Parameters of that transform.
    :return: Callable mapping a potency sum to an activity.
    :raises ValueError: If the name is unknown or the parameters do not match it.
    """

    activity_transform_registry.validate(name, parameters)
    return activity_transform_registry.factory(name)(**parameters)


def activity_budget(
    times: np.ndarray | Sequence[float],
    activities: np.ndarray | Sequence[float],
) -> float:
    r"""Integrate an activity trajectory into a budget.

    .. math::
        B(T) = \int_0^T A(t)\,\mathrm{d}t

    The integral is taken by the trapezoidal rule over the supplied grid, so the
    caller controls the time resolution and the horizon.

    :param times: Strictly increasing time grid, shape ``(n_steps,)``.
    :param activities: Activity at each time, shape ``(n_steps,)``.
    :return: The integrated budget, in units of activity multiplied by time.
    :raises ValueError: If the grid is not strictly increasing or the shapes
        disagree.
    """

    time_array = np.asarray(times, dtype=np.float64)
    activity_array = np.asarray(activities, dtype=np.float64)
    if time_array.shape != activity_array.shape:
        raise ValueError("times and activities must have the same shape.")
    if time_array.size < 2:
        raise ValueError("at least two time points are required.")
    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    ## REMARK: numpy renamed trapz to trapezoid in 2.0; the project pins 1.26, so
    ## the older name is used where the new one is absent.
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(activity_array, time_array))
