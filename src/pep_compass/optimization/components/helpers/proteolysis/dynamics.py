r"""Proteolysis in time: a Markov jump process over the fragments of one peptide.

## The process

A peptide exposed to a protease panel is a continuous-time Markov chain. The
state is the set of fragments the molecule has been reduced to. Every admissible
(fragment, bond, protease) triple is an independent Poisson clock with intensity
:math:`\lambda`, so the waiting time to the next event anywhere is exponential
with the summed intensity

.. math::
    \Lambda(S) = \sum_{f \in S} \sum_{\pi} \sum_{b \in \mathcal{B}(f, \pi)} \lambda_{\pi, b},

and the event that fires is drawn in proportion to its own intensity. That is the
Gillespie construction, and it is **exact**: it produces the event times of the
process itself, not a discretisation of them. No time step is chosen, so no time
step can be too coarse.

## Geometry is enforced, not assumed

:math:`\mathcal{B}(f, \pi)` comes from the event geometry of the protease. An
endopeptidase contributes every internal bond of the fragment. An exopeptidase
contributes the single bond its mechanism allows and **never an internal one**;
an internal position is not a slow cut for it, it is not a cut at all. The
distinction is made here by construction rather than by a weight, so an
exopeptidase cannot leak into the internal-cut channel.

## What an event does to the state

An endopeptidase cut replaces one fragment by the two pieces it separates. An
exopeptidase event also replaces one fragment by two, but one of them is the
released residue, dipeptide or tripeptide. Both are kept, so the fragments always
partition the parent sequence and nothing is lost from the bookkeeping. The
released piece is usually inactive, which the activity model discovers from its
predicted MIC rather than being told.

## The intensity

.. math::
    \lambda_{\pi, b} = w_\pi \exp\left(\alpha + \beta\, S^{M}_{\pi, b}\right)

``w`` is a per-protease weight, either uniform or the circulating amount;
:math:`\alpha` sets the global time unit and is calibrated elsewhere;
:math:`\beta` converts the window score into a log-rate. The separation matters:
:math:`\alpha` changes only the unit of time and never the ordering of events,
whereas :math:`\beta` changes which bond is cut first.

## The degenerate case

If every intensity is equal, the process reduces to a uniform random cut and the
specificity model contributes nothing. That is not a defect of the construction
but a property of the matrices: it happens exactly when they carry no information
above their sampling floor. :func:`hazard_concentration` measures how far a given
panel is from that case, so a run can report whether its result rests on
specificity or on geometry alone.

Dimension symbols: ``N`` proteases, ``P`` subsites (8), ``A`` amino acids (20),
``L`` residues of the parent peptide.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from pep_compass.optimization.components.helpers.proteolysis.geometry import (
    EventGeometry,
    allowed_cut_positions,
    masked_window_score,
)


class Fragment(NamedTuple):
    """A contiguous piece of the parent peptide.

    Positions are zero-based and half-open, so ``Fragment(0, len(sequence))`` is
    the intact peptide and the fragments of a state always partition it.
    """

    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the number of residues in the fragment."""

        return self.end - self.start

    def sequence(self, parent: str) -> str:
        """Return the residues of this fragment.

        :param parent: Parent peptide sequence.
        :return: The fragment as a string.
        """

        return parent[self.start : self.end]


class CleavageEvent(NamedTuple):
    """One admissible cleavage, with the intensity of its Poisson clock."""

    fragment_index: int
    protease_index: int
    bond: int
    intensity: float


@dataclass(frozen=True, slots=True)
class ProteaseKinetics:
    """The panel as the dynamics needs it: matrices, geometries and weights.

    :param matrices: Specificity log-probabilities, shape ``(N, P, A)``.
    :param geometries: Event geometry of each protease, length ``N``.
    :param weights: Per-protease weight :math:`w_\\pi`, shape ``(N,)``. Either
        uniform or the circulating amount, depending on the variant being run.
    :param codes: MEROPS codes, one per protease, for reporting.
    :param intercept: Global log-intensity offset :math:`\\alpha`, which sets the
        unit of time.
    :param slope: Coefficient :math:`\\beta` converting a window score into a
        log-intensity.
    """

    matrices: np.ndarray
    geometries: tuple[EventGeometry, ...]
    weights: np.ndarray
    codes: tuple[str, ...]
    intercept: float = 0.0
    slope: float = 1.0

    def __post_init__(self) -> None:
        if self.matrices.shape[0] != len(self.geometries):
            raise ValueError("one geometry is required per protease matrix.")
        if self.matrices.shape[0] != self.weights.shape[0]:
            raise ValueError("one weight is required per protease matrix.")
        if self.matrices.shape[0] != len(self.codes):
            raise ValueError("one code is required per protease matrix.")
        if np.any(self.weights < 0.0):
            raise ValueError("weights cannot be negative.")


def fragment_events(
    parent: str,
    fragments: Sequence[Fragment],
    kinetics: ProteaseKinetics,
    background: np.ndarray | None = None,
) -> list[CleavageEvent]:
    r"""Enumerate every admissible cleavage of a state, with its intensity.

    :param parent: Parent peptide sequence.
    :param fragments: Fragments making up the current state.
    :param kinetics: Panel matrices, geometries and weights.
    :param background: Background residue distribution ``(A,)``, uniform by
        default.
    :return: One entry per admissible (fragment, protease, bond) triple. Proteases
        with zero weight and fragments too short to cut contribute nothing.
    """

    events: list[CleavageEvent] = []
    for fragment_index, fragment in enumerate(fragments):
        if fragment.length < 2:
            continue
        sequence = fragment.sequence(parent)
        for protease_index, geometry in enumerate(kinetics.geometries):
            weight = float(kinetics.weights[protease_index])
            if weight <= 0.0:
                continue
            admissible = allowed_cut_positions(fragment.length, geometry)
            for bond in np.flatnonzero(admissible):
                score, _ = masked_window_score(
                    sequence, int(bond), kinetics.matrices[protease_index], background=background
                )
                intensity = weight * float(
                    np.exp(kinetics.intercept + kinetics.slope * score)
                )
                events.append(
                    CleavageEvent(fragment_index, protease_index, int(bond), intensity)
                )
    return events


def apply_event(fragments: Sequence[Fragment], event: CleavageEvent) -> tuple[Fragment, ...]:
    """Return the state after one cleavage.

    The cut fragment is replaced by the two pieces it separates. Both are kept,
    including the short piece an exopeptidase releases, so the fragments continue
    to partition the parent.

    :param fragments: Fragments making up the current state.
    :param event: The cleavage that fired.
    :return: The fragments of the resulting state.
    :raises ValueError: If the event does not name a bond of its fragment.
    """

    target = fragments[event.fragment_index]
    if not 0 <= event.bond < target.length - 1:
        raise ValueError(
            f"bond {event.bond} is not a bond of a fragment of length {target.length}."
        )
    cut_position = target.start + event.bond + 1
    remaining = [
        fragment for index, fragment in enumerate(fragments) if index != event.fragment_index
    ]
    return tuple(
        sorted(
            [*remaining, Fragment(target.start, cut_position), Fragment(cut_position, target.end)]
        )
    )


def hazard_concentration(events: Sequence[CleavageEvent]) -> float:
    r"""Measure how far a hazard profile is from a uniform one.

    The value is the normalized entropy of the intensities: one when every
    admissible event is equally likely, and approaching zero when a single event
    carries the whole hazard.

    A value near one says the specificity model is contributing nothing and the
    process is a uniform random cut with extra steps. Reporting it alongside a
    result states plainly whether the result rests on specificity.

    :param events: Admissible events of a state.
    :return: Normalized entropy in ``[0, 1]``, or ``0.0`` when there is at most
        one event.
    """

    if len(events) < 2:
        return 0.0
    intensities = np.array([event.intensity for event in events], dtype=np.float64)
    total = intensities.sum()
    if total <= 0.0:
        return 0.0
    probabilities = intensities / total
    positive = probabilities[probabilities > 0.0]
    entropy = -float((positive * np.log(positive)).sum())
    return entropy / float(np.log(len(events)))


class TrajectoryStep(NamedTuple):
    """One state of a simulated trajectory, valid from ``time`` onwards."""

    time: float
    fragments: tuple[Fragment, ...]
    total_hazard: float
    hazard_concentration: float


def simulate_trajectory(
    parent: str,
    kinetics: ProteaseKinetics,
    generator: np.random.Generator,
    horizon: float,
    background: np.ndarray | None = None,
    max_events: int = 1000,
) -> list[TrajectoryStep]:
    r"""Simulate one molecule by the Gillespie construction.

    The waiting time to the next event is drawn from
    :math:`\mathrm{Exp}(\Lambda)` and the event that fires is chosen in
    proportion to its intensity. Event times are exact; nothing is discretised.

    Because cleavage is first order in the peptide, molecules are independent, so
    a population is the distribution over repeated single-molecule trajectories.

    :param parent: Parent peptide sequence.
    :param kinetics: Panel matrices, geometries and weights.
    :param generator: Random generator.
    :param horizon: Time after which the simulation stops.
    :param background: Background residue distribution ``(A,)``.
    :param max_events: Guard against a pathological rate profile.
    :return: The state after each event, starting from the intact peptide at
        time zero.
    """

    fragments: tuple[Fragment, ...] = (Fragment(0, len(parent)),)
    time = 0.0
    events = fragment_events(parent, fragments, kinetics, background)
    total = float(sum(event.intensity for event in events))
    steps = [TrajectoryStep(0.0, fragments, total, hazard_concentration(events))]

    for _ in range(max_events):
        if not events or total <= 0.0:
            break
        time += float(generator.exponential(1.0 / total))
        if time > horizon:
            break
        ## The firing event is drawn in proportion to its own intensity.
        intensities = np.array([event.intensity for event in events], dtype=np.float64)
        chosen = events[int(generator.choice(len(events), p=intensities / total))]
        fragments = apply_event(fragments, chosen)
        events = fragment_events(parent, fragments, kinetics, background)
        total = float(sum(event.intensity for event in events))
        steps.append(TrajectoryStep(time, fragments, total, hazard_concentration(events)))
    return steps


def state_at(steps: Sequence[TrajectoryStep], time: float) -> tuple[Fragment, ...]:
    """Return the fragments a trajectory is in at a given time.

    :param steps: Output of :func:`simulate_trajectory`.
    :param time: Time to evaluate.
    :return: The fragments of the state holding at that time.
    :raises ValueError: If the time precedes the start of the trajectory.
    """

    if time < steps[0].time:
        raise ValueError("time precedes the start of the trajectory.")
    index = int(np.searchsorted([step.time for step in steps], time, side="right")) - 1
    return steps[index].fragments


def first_cut_distribution(
    parent: str,
    kinetics: ProteaseKinetics,
    background: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Return the exact distribution of the first cleavage of an intact peptide.

    The first event needs no simulation: competing exponential clocks give the
    probability of each event as its share of the total intensity, and the
    expected waiting time as :math:`1/\Lambda`. This is the single-cut analysis,
    and it is exact.

    :param parent: Parent peptide sequence.
    :param kinetics: Panel matrices, geometries and weights.
    :param background: Background residue distribution ``(A,)``.
    :return: Tuple of the probability that the first cut falls at each bond,
        shape ``(L - 1,)``, the share of the hazard contributed by each protease,
        shape ``(N,)``, and the expected time to the first event.
    """

    events = fragment_events(parent, (Fragment(0, len(parent)),), kinetics, background)
    n_bonds = max(len(parent) - 1, 0)
    bond_probability = np.zeros(n_bonds)  # (L - 1,)
    protease_share = np.zeros(len(kinetics.codes))  # (N,)
    total = float(sum(event.intensity for event in events))
    if total <= 0.0:
        return bond_probability, protease_share, float("inf")

    for event in events:
        bond_probability[event.bond] += event.intensity / total
        protease_share[event.protease_index] += event.intensity / total
    return bond_probability, protease_share, 1.0 / total
