"""Tests for the proteolysis Markov jump process."""

from __future__ import annotations

import numpy as np
import pytest

from pep_compass.optimization.components.helpers.proteolysis import (
    MEROPS_AMINO_ACIDS,
    MEROPS_SUBSITES,
)
from pep_compass.optimization.components.helpers.proteolysis.dynamics import (
    Fragment,
    ProteaseKinetics,
    apply_event,
    first_cut_distribution,
    fragment_events,
    hazard_concentration,
    simulate_trajectory,
    state_at,
)
from pep_compass.optimization.components.helpers.proteolysis.geometry import EventGeometry


N_SUBSITES = len(MEROPS_SUBSITES)
N_RESIDUES = len(MEROPS_AMINO_ACIDS)
PARENT = "AAAAKAAAAKAAAA"  # 14 residues, lysine at positions 4 and 9


def _uniform_matrix() -> np.ndarray:
    """Build a matrix with no residue preference.

    :return: Log-probabilities equal at every subsite, shape ``(P, A)``.
    """

    return np.full((N_SUBSITES, N_RESIDUES), np.log(1.0 / N_RESIDUES))


def _matrix_favouring(residue: str, subsite: int) -> np.ndarray:
    """Build a matrix preferring one residue at one subsite.

    :param residue: Preferred residue.
    :param subsite: Subsite carrying the preference.
    :return: Log-probabilities, shape ``(P, A)``.
    """

    probabilities = np.full((N_SUBSITES, N_RESIDUES), 1.0 / N_RESIDUES)
    column = MEROPS_AMINO_ACIDS.index(residue)
    probabilities[subsite] = 0.01 / (N_RESIDUES - 1)
    probabilities[subsite, column] = 0.99
    return np.log(probabilities)


def _kinetics(
    matrices: list[np.ndarray],
    geometries: list[EventGeometry],
    weights: list[float] | None = None,
    intercept: float = 0.0,
    slope: float = 1.0,
) -> ProteaseKinetics:
    """Assemble a panel for the tests.

    :param matrices: One specificity matrix per protease.
    :param geometries: One geometry per protease.
    :param weights: Per-protease weights, ones by default.
    :param intercept: Global log-intensity offset.
    :param slope: Score-to-log-intensity coefficient.
    :return: The assembled panel.
    """

    return ProteaseKinetics(
        matrices=np.stack(matrices),
        geometries=tuple(geometries),
        weights=np.array(weights if weights is not None else [1.0] * len(matrices)),
        codes=tuple(f"X{index:02d}.001" for index in range(len(matrices))),
        intercept=intercept,
        slope=slope,
    )


class TestGeometryIsEnforced:
    """An exopeptidase must never reach an internal bond."""

    def test_endopeptidase_offers_every_internal_bond(self) -> None:
        """All bonds of the intact peptide are admissible for an endopeptidase."""

        kinetics = _kinetics([_uniform_matrix()], [EventGeometry.ENDOPEPTIDASE])
        events = fragment_events(PARENT, (Fragment(0, len(PARENT)),), kinetics)
        assert {event.bond for event in events} == set(range(len(PARENT) - 1))

    @pytest.mark.parametrize(
        ("geometry", "expected_bond"),
        [
            (EventGeometry.AMINOPEPTIDASE, 0),
            (EventGeometry.DIPEPTIDYL_PEPTIDASE, 1),
            (EventGeometry.CARBOXYPEPTIDASE, len(PARENT) - 2),
            (EventGeometry.PEPTIDYL_DIPEPTIDASE, len(PARENT) - 3),
        ],
    )
    def test_exopeptidase_offers_exactly_one_bond(
        self, geometry: EventGeometry, expected_bond: int
    ) -> None:
        """The single admissible bond is the one the mechanism allows.

        This is the property that keeps an exopeptidase out of the internal-cut
        channel entirely, rather than merely making internal cuts unlikely.
        """

        kinetics = _kinetics([_uniform_matrix()], [geometry])
        events = fragment_events(PARENT, (Fragment(0, len(PARENT)),), kinetics)
        assert len(events) == 1
        assert events[0].bond == expected_bond

    def test_unresolved_geometry_produces_no_event(self) -> None:
        """A protease whose geometry is unknown is never simulated."""

        kinetics = _kinetics([_uniform_matrix()], [EventGeometry.UNKNOWN])
        assert fragment_events(PARENT, (Fragment(0, len(PARENT)),), kinetics) == []

    def test_zero_weight_removes_a_protease(self) -> None:
        """A protease absent from the matrix contributes no hazard."""

        kinetics = _kinetics(
            [_uniform_matrix(), _uniform_matrix()],
            [EventGeometry.ENDOPEPTIDASE, EventGeometry.ENDOPEPTIDASE],
            weights=[1.0, 0.0],
        )
        events = fragment_events(PARENT, (Fragment(0, len(PARENT)),), kinetics)
        assert {event.protease_index for event in events} == {0}


class TestStateTransitions:
    """What an event does to the fragments."""

    def test_an_internal_cut_splits_one_fragment_into_two(self) -> None:
        """The cut fragment is replaced by the two pieces it separates."""

        kinetics = _kinetics([_uniform_matrix()], [EventGeometry.ENDOPEPTIDASE])
        events = fragment_events(PARENT, (Fragment(0, len(PARENT)),), kinetics)
        event = next(item for item in events if item.bond == 5)
        fragments = apply_event((Fragment(0, len(PARENT)),), event)
        assert fragments == (Fragment(0, 6), Fragment(6, len(PARENT)))

    def test_fragments_always_partition_the_parent(self) -> None:
        """Nothing is lost from the bookkeeping, including released residues.

        An aminopeptidase releases a single residue, which is kept as a fragment
        so that the union of the state remains the whole parent.
        """

        kinetics = _kinetics(
            [_uniform_matrix()], [EventGeometry.AMINOPEPTIDASE], intercept=2.0
        )
        steps = simulate_trajectory(
            PARENT, kinetics, np.random.default_rng(1), horizon=100.0
        )
        for step in steps:
            covered = sorted((fragment.start, fragment.end) for fragment in step.fragments)
            assert covered[0][0] == 0
            assert covered[-1][1] == len(PARENT)
            for left, right in zip(covered, covered[1:]):
                assert left[1] == right[0]

    def test_aminopeptidase_shortens_by_one_and_can_act_again(self) -> None:
        """Trimming is a ladder, not a single event."""

        kinetics = _kinetics(
            [_uniform_matrix()], [EventGeometry.AMINOPEPTIDASE], intercept=3.0
        )
        steps = simulate_trajectory(
            PARENT, kinetics, np.random.default_rng(0), horizon=100.0
        )
        assert len(steps) > 3, "the enzyme must be able to attack its own product"
        longest = [max(fragment.length for fragment in step.fragments) for step in steps]
        assert longest == sorted(longest, reverse=True)
        assert all(
            earlier - later in (0, 1) for earlier, later in zip(longest, longest[1:])
        )

    def test_an_invalid_bond_is_rejected(self) -> None:
        """An event naming a bond its fragment does not have raises."""

        from pep_compass.optimization.components.helpers.proteolysis.dynamics import CleavageEvent

        with pytest.raises(ValueError):
            apply_event((Fragment(0, 3),), CleavageEvent(0, 0, 5, 1.0))


class TestFirstCutDistribution:
    """The exact single-cut analysis."""

    def test_probabilities_sum_to_one(self) -> None:
        """Competing clocks give a proper distribution over bonds."""

        kinetics = _kinetics([_matrix_favouring("K", 3)], [EventGeometry.ENDOPEPTIDASE])
        bonds, proteases, expected_time = first_cut_distribution(PARENT, kinetics)
        assert bonds.sum() == pytest.approx(1.0)
        assert proteases.sum() == pytest.approx(1.0)
        assert expected_time > 0.0

    def test_the_preferred_bond_is_the_most_likely(self) -> None:
        """A matrix preferring lysine at P1 cuts after a lysine."""

        kinetics = _kinetics([_matrix_favouring("K", 3)], [EventGeometry.ENDOPEPTIDASE])
        bonds, _, _ = first_cut_distribution(PARENT, kinetics)
        ## P1 is the residue before the cut, and the lysines sit at index 4 and 9.
        assert int(np.argmax(bonds)) in {4, 9}

    def test_the_intercept_scales_time_without_moving_the_ordering(self) -> None:
        """The global offset sets the unit of time and nothing else."""

        slow = _kinetics([_matrix_favouring("K", 3)], [EventGeometry.ENDOPEPTIDASE], intercept=0.0)
        fast = _kinetics([_matrix_favouring("K", 3)], [EventGeometry.ENDOPEPTIDASE], intercept=1.0)
        slow_bonds, _, slow_time = first_cut_distribution(PARENT, slow)
        fast_bonds, _, fast_time = first_cut_distribution(PARENT, fast)
        assert np.allclose(slow_bonds, fast_bonds)
        assert fast_time == pytest.approx(slow_time / np.e)

    def test_the_slope_changes_which_bond_is_cut_first(self) -> None:
        """Unlike the intercept, the score coefficient reorders the events."""

        flat = _kinetics([_matrix_favouring("K", 3)], [EventGeometry.ENDOPEPTIDASE], slope=0.0)
        steep = _kinetics([_matrix_favouring("K", 3)], [EventGeometry.ENDOPEPTIDASE], slope=2.0)
        flat_bonds, _, _ = first_cut_distribution(PARENT, flat)
        steep_bonds, _, _ = first_cut_distribution(PARENT, steep)
        assert flat_bonds.max() == pytest.approx(flat_bonds.min())
        assert steep_bonds.max() > flat_bonds.max()

    def test_protease_share_follows_the_weights(self) -> None:
        """A protease present at twice the amount carries twice the hazard."""

        kinetics = _kinetics(
            [_uniform_matrix(), _uniform_matrix()],
            [EventGeometry.ENDOPEPTIDASE, EventGeometry.ENDOPEPTIDASE],
            weights=[1.0, 2.0],
        )
        _, shares, _ = first_cut_distribution(PARENT, kinetics)
        assert shares[1] == pytest.approx(2.0 * shares[0])


class TestHazardConcentration:
    """Whether a result rests on specificity or on geometry alone."""

    def test_a_uniform_matrix_gives_the_maximum(self) -> None:
        """Equal intensities mean the cut position carries no information."""

        kinetics = _kinetics([_uniform_matrix()], [EventGeometry.ENDOPEPTIDASE])
        events = fragment_events(PARENT, (Fragment(0, len(PARENT)),), kinetics)
        assert hazard_concentration(events) == pytest.approx(1.0)

    def test_a_specific_matrix_gives_less(self) -> None:
        """A matrix that prefers one context concentrates the hazard."""

        kinetics = _kinetics([_matrix_favouring("K", 3)], [EventGeometry.ENDOPEPTIDASE], slope=2.0)
        events = fragment_events(PARENT, (Fragment(0, len(PARENT)),), kinetics)
        assert hazard_concentration(events) < 0.95

    def test_a_single_event_is_reported_as_zero(self) -> None:
        """One admissible event carries no distribution to measure."""

        kinetics = _kinetics([_uniform_matrix()], [EventGeometry.AMINOPEPTIDASE])
        events = fragment_events(PARENT, (Fragment(0, len(PARENT)),), kinetics)
        assert hazard_concentration(events) == 0.0


class TestSimulation:
    """Properties of the simulated trajectories."""

    def test_the_first_event_time_is_exponential_with_the_total_hazard(self) -> None:
        """The construction reproduces the waiting-time distribution it assumes."""

        kinetics = _kinetics([_matrix_favouring("K", 3)], [EventGeometry.ENDOPEPTIDASE])
        _, _, expected_time = first_cut_distribution(PARENT, kinetics)
        generator = np.random.default_rng(3)
        first_times = []
        for _ in range(600):
            steps = simulate_trajectory(PARENT, kinetics, generator, horizon=1e6)
            if len(steps) > 1:
                first_times.append(steps[1].time)
        assert np.mean(first_times) == pytest.approx(expected_time, rel=0.15)

    def test_a_horizon_truncates_the_trajectory(self) -> None:
        """No event is recorded after the horizon."""

        kinetics = _kinetics([_uniform_matrix()], [EventGeometry.ENDOPEPTIDASE], intercept=2.0)
        steps = simulate_trajectory(
            PARENT, kinetics, np.random.default_rng(5), horizon=0.5
        )
        assert all(step.time <= 0.5 for step in steps)

    def test_the_trajectory_terminates_when_nothing_can_be_cut(self) -> None:
        """Fragments shorter than two residues offer no bond."""

        kinetics = _kinetics([_uniform_matrix()], [EventGeometry.ENDOPEPTIDASE], intercept=5.0)
        steps = simulate_trajectory(
            "AAAA", kinetics, np.random.default_rng(7), horizon=1e6, max_events=200
        )
        assert all(fragment.length == 1 for fragment in steps[-1].fragments)

    def test_state_at_returns_the_state_holding_at_a_time(self) -> None:
        """The state is right-continuous in the event times."""

        kinetics = _kinetics([_uniform_matrix()], [EventGeometry.ENDOPEPTIDASE], intercept=1.0)
        steps = simulate_trajectory(PARENT, kinetics, np.random.default_rng(11), horizon=10.0)
        assert state_at(steps, 0.0) == steps[0].fragments
        if len(steps) > 1:
            just_before = steps[1].time * (1 - 1e-9)
            assert state_at(steps, just_before) == steps[0].fragments
            assert state_at(steps, steps[1].time) == steps[1].fragments

    def test_a_time_before_the_start_is_rejected(self) -> None:
        """A negative time has no state."""

        kinetics = _kinetics([_uniform_matrix()], [EventGeometry.ENDOPEPTIDASE])
        steps = simulate_trajectory(PARENT, kinetics, np.random.default_rng(13), horizon=1.0)
        with pytest.raises(ValueError):
            state_at(steps, -1.0)

    def test_endopeptidase_and_exopeptidase_reach_different_states(self) -> None:
        """The two geometries produce different fragment ladders.

        An exopeptidase leaves one long fragment and a trail of short released
        pieces; an endopeptidase splits the peptide into comparable halves. The
        distinction is what makes their consequence for activity different.
        """

        endo = _kinetics([_uniform_matrix()], [EventGeometry.ENDOPEPTIDASE], intercept=2.0)
        exo = _kinetics([_uniform_matrix()], [EventGeometry.AMINOPEPTIDASE], intercept=2.0)
        endo_steps = simulate_trajectory(PARENT, endo, np.random.default_rng(17), horizon=1.0)
        exo_steps = simulate_trajectory(PARENT, exo, np.random.default_rng(17), horizon=1.0)
        exo_released = [
            fragment for fragment in exo_steps[-1].fragments if fragment.length == 1
        ]
        assert len(exo_released) == len(exo_steps) - 1
        assert max(fragment.length for fragment in exo_steps[-1].fragments) >= len(PARENT) - len(
            exo_steps
        )
        assert len(endo_steps[-1].fragments) == len(endo_steps)


class TestPanelValidation:
    """The panel refuses to be assembled inconsistently."""

    @pytest.mark.parametrize("field", ["geometries", "weights", "codes"])
    def test_mismatched_lengths_are_rejected(self, field: str) -> None:
        """One geometry, weight and code is required per matrix."""

        arguments = {
            "matrices": np.stack([_uniform_matrix(), _uniform_matrix()]),
            "geometries": (EventGeometry.ENDOPEPTIDASE, EventGeometry.ENDOPEPTIDASE),
            "weights": np.array([1.0, 1.0]),
            "codes": ("A01.001", "A01.002"),
        }
        arguments[field] = arguments[field][:1]
        with pytest.raises(ValueError):
            ProteaseKinetics(**arguments)

    def test_negative_weights_are_rejected(self) -> None:
        """A negative amount has no meaning as an intensity multiplier."""

        with pytest.raises(ValueError):
            _kinetics([_uniform_matrix()], [EventGeometry.ENDOPEPTIDASE], weights=[-1.0])
