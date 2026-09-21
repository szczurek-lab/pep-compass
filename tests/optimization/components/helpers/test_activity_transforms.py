"""Tests for the MIC-to-activity transforms."""

from __future__ import annotations

import numpy as np
import pytest

from pep_compass.optimization.components.helpers.activity import (
    DEFAULT_GROWTH_RATE,
    DEFAULT_MAX_KILL_RATE,
    activity_budget,
    activity_transform_registry,
    build_activity_transform,
    hill_kill_rate,
    log_potency,
    net_kill_rate,
    potency_sum,
    soft_threshold,
    threshold,
)


class TestPotencySum:
    """The state variable summarizing a mixture of species."""

    def test_single_species_is_the_concentration_over_its_mic(self) -> None:
        """One species reduces to the plain potency ratio."""

        assert potency_sum([4.0], [2.0]) == pytest.approx(2.0)

    def test_species_contribute_additively(self) -> None:
        """A cut replaces one term by two and changes nothing else.

        This is the property that makes the quantity usable as the state of a
        cleavage process.
        """

        parent = potency_sum([1.0], [4.0])
        fragments = potency_sum([1.0, 1.0], [8.0, 8.0])
        assert fragments == pytest.approx(parent)

    def test_an_inactive_fragment_contributes_almost_nothing(self) -> None:
        """A fragment with a very high MIC barely moves the state."""

        with_inactive = potency_sum([1.0, 1.0], [4.0, 1e6])
        without = potency_sum([1.0], [4.0])
        assert with_inactive == pytest.approx(without, rel=1e-4)

    def test_batches_are_reduced_over_the_last_axis(self) -> None:
        """A batch of mixtures yields one value per mixture."""

        result = potency_sum([[1.0, 1.0], [2.0, 0.0]], [[2.0, 2.0], [4.0, 4.0]])
        assert result.shape == (2,)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.5)

    def test_non_positive_mic_is_rejected(self) -> None:
        """A MIC of zero would make the ratio undefined."""

        with pytest.raises(ValueError):
            potency_sum([1.0], [0.0])


class TestRegistry:
    """Selecting a transform by name."""

    def test_every_transform_is_registered(self) -> None:
        """The registry is the single list of available constructions."""

        assert set(activity_transform_registry.names()) == {
            "hill_kill_rate",
            "log_potency",
            "net_kill_rate",
            "potency_sum",
            "soft_threshold",
            "threshold",
        }

    def test_building_by_name_matches_calling_the_factory(self) -> None:
        """The registry returns the same transform the factory would."""

        by_name = build_activity_transform("hill_kill_rate", hill_coefficient=2.0)
        direct = hill_kill_rate(hill_coefficient=2.0)
        grid = np.array([0.1, 1.0, 10.0])
        assert np.allclose(by_name(grid), direct(grid))

    def test_unknown_parameter_is_rejected(self) -> None:
        """A misspelled parameter fails at construction, not silently."""

        with pytest.raises(ValueError):
            build_activity_transform("hill_kill_rate", steepness=2.0)

    def test_unknown_name_is_rejected(self) -> None:
        """An unregistered transform cannot be built."""

        with pytest.raises(ValueError):
            build_activity_transform("not_a_transform")


class TestHillKillRate:
    """The saturating pharmacodynamic rate."""

    def test_half_maximum_sits_at_the_inhibitory_boundary(self) -> None:
        """The rate is half its maximum at the MIC, for any steepness."""

        for coefficient in (0.5, 1.0, 4.0):
            transform = hill_kill_rate(hill_coefficient=coefficient, max_kill_rate=2.0)
            assert float(transform(np.array(1.0))) == pytest.approx(1.0)

    def test_lower_mic_kills_faster(self) -> None:
        """A more potent mixture has a strictly higher rate."""

        transform = hill_kill_rate()
        weak = potency_sum([1.0], [8.0])
        strong = potency_sum([1.0], [2.0])
        assert float(transform(strong)) > float(transform(weak))

    def test_the_rate_saturates(self) -> None:
        """No mixture kills faster than the maximum rate.

        Without saturation a single fragment with an extreme predicted MIC would
        dominate the integrated budget however briefly it existed.
        """

        transform = hill_kill_rate(max_kill_rate=3.0)
        assert float(transform(np.array(1e9))) == pytest.approx(3.0, rel=1e-6)
        assert float(transform(np.array(1e9))) < 3.0

    def test_steeper_hill_sharpens_the_transition(self) -> None:
        """A larger coefficient approaches a step at the boundary."""

        shallow = hill_kill_rate(hill_coefficient=1.0)
        steep = hill_kill_rate(hill_coefficient=8.0)
        assert float(steep(np.array(0.5))) < float(shallow(np.array(0.5)))
        assert float(steep(np.array(2.0))) > float(shallow(np.array(2.0)))

    def test_non_positive_parameters_are_rejected(self) -> None:
        """A zero maximum rate would make every mixture inert."""

        with pytest.raises(ValueError):
            hill_kill_rate(max_kill_rate=0.0)


class TestNetKillRate:
    """The rate net of bacterial regrowth."""

    def test_default_growth_places_the_zero_at_the_mic(self) -> None:
        """With the default growth rate, net activity vanishes at the MIC.

        A MIC is by definition the concentration at which growth is just
        inhibited, so this is the transform agreeing with its own input.
        """

        for coefficient in (0.5, 1.0, 4.0):
            transform = net_kill_rate(hill_coefficient=coefficient)
            assert float(transform(np.array(1.0))) == pytest.approx(0.0, abs=1e-12)

    def test_activity_turns_negative_below_the_boundary(self) -> None:
        """A degraded mixture loses ground instead of standing still."""

        transform = net_kill_rate()
        assert float(transform(np.array(0.25))) < 0.0
        assert float(transform(np.array(4.0))) > 0.0

    def test_growth_faster_than_killing_is_rejected(self) -> None:
        """If growth exceeds the maximum rate, no mixture can ever kill."""

        with pytest.raises(ValueError):
            net_kill_rate(max_kill_rate=1.0, growth_rate=1.0)

    def test_balance_point_matches_the_closed_form(self) -> None:
        """The zero of the net rate is where the closed form puts it."""

        coefficient, maximum, growth = 2.0, 1.0, 0.2
        transform = net_kill_rate(coefficient, maximum, growth)
        balance = (growth / (maximum - growth)) ** (1.0 / coefficient)
        assert float(transform(np.array(balance))) == pytest.approx(0.0, abs=1e-12)


class TestThresholdTransforms:
    """The binary and smoothed indicators of inhibition."""

    def test_threshold_is_an_indicator_of_the_boundary(self) -> None:
        """Below the boundary the mixture scores zero, at or above it one."""

        transform = threshold()
        assert float(transform(np.array(0.999))) == 0.0
        assert float(transform(np.array(1.0))) == 1.0

    def test_threshold_discards_magnitude(self) -> None:
        """Two mixtures differing a hundredfold above the boundary score alike."""

        transform = threshold()
        assert float(transform(np.array(1.5))) == float(transform(np.array(150.0)))

    def test_soft_threshold_is_half_at_the_boundary(self) -> None:
        """The smoothed indicator crosses its midpoint at the boundary."""

        assert float(soft_threshold()(np.array(1.0))) == pytest.approx(0.5)

    def test_soft_threshold_width_is_in_dilution_steps(self) -> None:
        """One doubling of potency moves a unit-width transform by a fixed step."""

        transform = soft_threshold(width_in_doublings=1.0)
        assert float(transform(np.array(2.0))) == pytest.approx(2.0 / 3.0)
        assert float(transform(np.array(0.5))) == pytest.approx(1.0 / 3.0)

    def test_narrow_soft_threshold_approaches_the_hard_one(self) -> None:
        """Shrinking the width recovers the binary indicator."""

        narrow = soft_threshold(width_in_doublings=0.01)
        assert float(narrow(np.array(1.2))) == pytest.approx(1.0, abs=1e-6)
        assert float(narrow(np.array(0.8))) == pytest.approx(0.0, abs=1e-6)


class TestLogPotency:
    """The logarithmic reading of potency."""

    def test_doubling_potency_adds_one(self) -> None:
        """The unit is the two-fold dilution step of the MIC assay."""

        transform = log_potency()
        assert float(transform(np.array(2.0))) - float(transform(np.array(1.0))) == pytest.approx(1.0)
        assert float(transform(np.array(8.0))) - float(transform(np.array(4.0))) == pytest.approx(1.0)

    def test_the_boundary_is_zero(self) -> None:
        """At the inhibitory boundary the logarithmic potency vanishes."""

        assert float(log_potency()(np.array(1.0))) == pytest.approx(0.0)

    def test_the_floor_bounds_the_value_from_below(self) -> None:
        """An entirely degraded mixture cannot produce minus infinity."""

        transform = log_potency(floor=1e-3)
        assert float(transform(np.array(0.0))) == pytest.approx(np.log2(1e-3))


class TestActivityBudget:
    """Integrating a trajectory."""

    def test_constant_activity_integrates_to_the_rectangle(self) -> None:
        """A flat trajectory gives rate multiplied by horizon."""

        times = np.linspace(0.0, 10.0, 11)
        assert activity_budget(times, np.full_like(times, 0.5)) == pytest.approx(5.0)

    def test_decaying_activity_accumulates_less(self) -> None:
        """A peptide losing activity earns a smaller budget."""

        times = np.linspace(0.0, 10.0, 101)
        assert activity_budget(times, np.exp(-times)) < activity_budget(
            times, np.full_like(times, 1.0)
        )

    def test_negative_activity_reduces_the_budget(self) -> None:
        """Once regrowth outpaces killing the budget falls.

        A trajectory decaying linearly from the maximum rate to its negative
        integrates to zero: the killing achieved early is exactly undone by the
        regrowth that follows.
        """

        times = np.linspace(0.0, 2.0, 21)
        activities = 1.0 - times
        assert activity_budget(times, activities) == pytest.approx(0.0, abs=1e-12)

        ## Stopping before the activity turns negative keeps the budget positive.
        early = times <= 1.0
        assert activity_budget(times[early], activities[early]) > 0.0

    def test_a_non_increasing_grid_is_rejected(self) -> None:
        """An unsorted time grid would silently give a wrong sign."""

        with pytest.raises(ValueError):
            activity_budget([0.0, 2.0, 1.0], [1.0, 1.0, 1.0])

    def test_mismatched_shapes_are_rejected(self) -> None:
        """The trajectory must be sampled on the supplied grid."""

        with pytest.raises(ValueError):
            activity_budget([0.0, 1.0], [1.0])


class TestTransformsAgreeOnOrdering:
    """Properties every construction must share."""

    @pytest.mark.parametrize(
        "name", ["hill_kill_rate", "net_kill_rate", "potency_sum", "log_potency", "soft_threshold"]
    )
    def test_activity_is_non_decreasing_in_potency(self, name: str) -> None:
        """A more potent mixture is never scored as less active.

        The binary threshold is excluded only because it is flat, not decreasing.
        """

        transform = build_activity_transform(name)
        grid = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
        values = transform(grid)
        assert np.all(np.diff(values) > 0.0)

    @pytest.mark.parametrize("name", ["hill_kill_rate", "net_kill_rate", "threshold", "soft_threshold"])
    def test_bounded_transforms_stay_bounded(self, name: str) -> None:
        """The saturating readings cannot exceed their maximum."""

        transform = build_activity_transform(name)
        assert float(transform(np.array(1e12))) <= DEFAULT_MAX_KILL_RATE + 1e-9
        assert float(transform(np.array(1e12))) >= -DEFAULT_GROWTH_RATE
