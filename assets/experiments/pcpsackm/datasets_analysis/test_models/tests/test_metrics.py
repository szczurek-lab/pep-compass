"""Tests of the scoring rules, with analytically known expectations."""

from __future__ import annotations

import numpy as np
import pytest

from model_benchmark.metrics import (
    baseline_metrics,
    classification_metrics,
    confusion_table,
    measurement_interval_metrics,
    regression_metrics,
)


def test_perfect_prediction_scores_exactly():
    """A perfect prediction has zero error, unit R2 and unit error factor."""
    true = np.array([2.0, 3.0, 4.0, 5.0])  # (4,)
    metrics = regression_metrics(true, true.copy())
    assert metrics["mae_log10"] == pytest.approx(0.0)
    assert metrics["rmse_log10"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["spearman_rho"] == pytest.approx(1.0)
    assert metrics["median_error_factor"] == pytest.approx(1.0)


def test_constant_offset_is_reported_as_bias_and_factor():
    """A fixed logarithmic offset appears as bias and as a multiplicative factor."""
    true = np.array([2.0, 3.0, 4.0, 5.0])  # (4,)
    metrics = regression_metrics(true, true + np.log10(2.0))
    assert metrics["bias_log10"] == pytest.approx(np.log10(2.0))
    assert metrics["median_error_factor"] == pytest.approx(2.0)
    ## REMARK: An offset preserves the ordering, so rank correlation stays
    ## perfect while R2 drops. Both must be read together.
    assert metrics["spearman_rho"] == pytest.approx(1.0)
    assert metrics["r2"] < 1.0


def test_worse_than_mean_gives_negative_r2():
    """Predicting the reverse order is worse than predicting the mean."""
    true = np.array([2.0, 3.0, 4.0, 5.0])  # (4,)
    metrics = regression_metrics(true, true[::-1].copy())
    assert metrics["r2"] < 0.0
    assert metrics["spearman_rho"] == pytest.approx(-1.0)


def test_baseline_is_the_constant_median():
    """The baseline reports the error of answering the cohort median."""
    true = np.array([1.0, 2.0, 3.0, 4.0])  # (4,)
    metrics = baseline_metrics(true)
    assert metrics["mae_log10"] == pytest.approx(np.abs(np.median(true) - true).mean())
    assert metrics["r2"] <= 0.0


def test_regression_metrics_reject_mismatched_input():
    """Shape mismatch and tiny samples are errors."""
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


class _FixedIntervalModel:
    """Interval estimator returning a fixed multiplicative band."""

    family = "test_fixed"
    domain_seconds = (1000.0, 20000.0)

    def predict(self, half_life_seconds, *, allow_extrapolation=False):
        """Return bounds a factor of two below and above the point.

        :param half_life_seconds: Point estimates.
        :param bool allow_extrapolation: Accepted for interface compatibility.
        :return: Lower and upper bounds.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        points = np.atleast_1d(np.asarray(half_life_seconds, dtype=float))  # (N,)
        return points / 2.0, points * 2.0


def test_interval_hit_rate_counts_predictions_inside_the_band(synthetic_cohort):
    """A prediction inside the band counts, one outside does not."""
    model = _FixedIntervalModel()
    measured = synthetic_cohort.log10_half_life.to_numpy(dtype=float)  # (N,)

    inside = measured + np.log10(1.5)
    metrics = measurement_interval_metrics(synthetic_cohort, inside, model)
    assert metrics["interval_hit_rate"] == pytest.approx(1.0)
    assert metrics["median_interval_factor"] == pytest.approx(4.0)

    outside = measured + np.log10(3.0)
    assert measurement_interval_metrics(synthetic_cohort, outside, model)["interval_hit_rate"] == pytest.approx(0.0)


def test_extrapolation_share_is_reported(synthetic_cohort):
    """Peptides outside the fitted domain are counted, not hidden."""
    metrics = measurement_interval_metrics(
        synthetic_cohort, synthetic_cohort.log10_half_life.to_numpy(dtype=float), _FixedIntervalModel()
    )
    ## REMARK: The synthetic cohort spans 600 s to 90000 s while the fixture
    ## domain is 1000 s to 20000 s, so three peptides lie outside.
    assert metrics["extrapolated_share"] == pytest.approx(3 / 6)


def test_class_metrics_use_the_contract_scheme(synthetic_cohort):
    """Predicting the measured value reproduces the measured classes exactly."""
    metrics = classification_metrics(
        synthetic_cohort, synthetic_cohort.log10_half_life.to_numpy(dtype=float)
    )
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["predicted_class_entropy_classes"] == 3


def test_class_metrics_detect_a_single_class_predictor(synthetic_cohort):
    """A predictor answering one value collapses to one class."""
    constant = np.full(len(synthetic_cohort), np.log10(1000.0))  # (N,)
    metrics = classification_metrics(synthetic_cohort, constant)
    assert metrics["predicted_class_entropy_classes"] == 1
    assert metrics["accuracy"] == pytest.approx(2 / 6)
    assert metrics["majority_class_accuracy"] == pytest.approx(2 / 6)


def test_confusion_table_is_square_and_ordered(synthetic_cohort):
    """The confusion matrix always carries all three classes in contract order."""
    table = confusion_table(synthetic_cohort, synthetic_cohort.log10_half_life.to_numpy(dtype=float))
    assert list(table.index) == ["short", "medium", "long"]
    assert list(table.columns) == ["short", "medium", "long"]
    assert int(table.to_numpy().sum()) == len(synthetic_cohort)
