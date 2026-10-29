"""Scoring rules for published half-life predictors.

Three questions are scored separately, from the hardest to the easiest:

1. point regression on ``log10`` seconds;
2. whether a prediction lands inside the measurement interval that this
   project estimates around a measured point, using the logarithmic
   relative-width estimator of the half-life analysis;
3. whether the predicted duration class matches the contract class.

Every metric is reported next to a trivial baseline, because a predictor that
cannot beat the cohort median is not useful regardless of its correlation.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from model_benchmark.config import CLASS_ORDER, DATA_MANAGER_SOURCE

logger = logging.getLogger(__name__)

if str(DATA_MANAGER_SOURCE) not in sys.path:
    sys.path.insert(0, str(DATA_MANAGER_SOURCE))

from proteolysis_data_manager.analysis.half_life.intervals import (  # noqa: E402
    fit_interval_model,
    reported_intervals,
)
from proteolysis_data_manager.integration.half_life.labels import point_class  # noqa: E402


def regression_metrics(true_log10, predicted_log10):
    """Score point predictions on the base-10 logarithmic half-life scale.

    :param numpy.ndarray true_log10: Measured ``log10`` seconds, shape ``(N,)``.
    :param numpy.ndarray predicted_log10: Predicted ``log10`` seconds, shape ``(N,)``.
    :return: Error, correlation and calibration metrics.
    :rtype: dict
    :raises ValueError: If the inputs differ in length or are shorter than three.
    """
    true = np.asarray(true_log10, dtype=float)  # (N,)
    predicted = np.asarray(predicted_log10, dtype=float)  # (N,)
    if true.shape != predicted.shape:
        raise ValueError("Measured and predicted arrays must have the same shape")
    if true.size < 3:
        raise ValueError("At least three peptides are required for regression metrics")

    residual = predicted - true  # (N,)
    total = float(((true - true.mean()) ** 2).sum())
    ## REMARK: R^2 is computed against the mean of the measured values, so a
    ## value below zero means the model is worse than predicting that mean.
    r_squared = 1.0 - float((residual**2).sum()) / total if total > 0 else float("nan")

    return {
        "peptides": int(true.size),
        "mae_log10": float(np.abs(residual).mean()),
        "rmse_log10": float(np.sqrt((residual**2).mean())),
        "median_absolute_log10": float(np.median(np.abs(residual))),
        "bias_log10": float(residual.mean()),
        "r2": r_squared,
        "pearson_r": float(pearsonr(true, predicted).statistic),
        "spearman_rho": float(spearmanr(true, predicted).statistic),
        ## REMARK: The multiplicative error factor is the plain reading of a
        ## logarithmic error: 10**MAE = 2 means "wrong by a factor of two".
        "median_error_factor": float(10.0 ** np.median(np.abs(residual))),
    }


def baseline_metrics(true_log10):
    """Score the constant predictor that always answers the cohort median.

    :param numpy.ndarray true_log10: Measured ``log10`` seconds, shape ``(N,)``.
    :return: Metrics of the trivial point baseline.
    :rtype: dict
    """
    true = np.asarray(true_log10, dtype=float)  # (N,)
    constant = np.full_like(true, np.median(true))  # (N,)
    metrics = {
        "peptides": int(true.size),
        "mae_log10": float(np.abs(constant - true).mean()),
        "rmse_log10": float(np.sqrt(((constant - true) ** 2).mean())),
        "median_absolute_log10": float(np.median(np.abs(constant - true))),
        "bias_log10": float((constant - true).mean()),
        "r2": float(1.0 - ((constant - true) ** 2).sum() / ((true - true.mean()) ** 2).sum()),
        "pearson_r": float("nan"),
        "spearman_rho": float("nan"),
        "median_error_factor": float(10.0 ** np.median(np.abs(constant - true))),
    }
    return metrics


def measurement_interval_model(family="logarithmic"):
    """Fit the project's point-to-interval estimator on the reported intervals.

    The estimator is fitted on the published Cavaco intervals, which are the
    only records in the dataset carrying confidence-interval endpoints.

    :param str family: Width family; ``logarithmic`` is the family selected by
        the half-life interval analysis notebook.
    :return: Fitted estimator and the records it was fitted on.
    :rtype: tuple[proteolysis_data_manager.analysis.half_life.intervals.IntervalModel, pandas.DataFrame]
    """
    from model_benchmark.dataset import load_tables

    experiments, sequences = load_tables()
    intervals = reported_intervals(experiments, sequences)
    model = fit_interval_model(intervals, family=family)
    logger.info(
        "Interval estimator (%s) fitted on %s records, domain %.0f-%.0f s",
        family,
        len(intervals),
        model.domain_seconds[0],
        model.domain_seconds[1],
    )
    return model, intervals


def measurement_interval_metrics(cohort, predicted_log10, interval_model):
    """Test whether predictions land inside the estimated measurement interval.

    For every measured point ``t`` the estimator returns bounds ``L(t)`` and
    ``U(t)``. A prediction counts as a hit when it lies between them.

    :param pandas.DataFrame cohort: Aggregated cohort with ``half_life_seconds``.
    :param numpy.ndarray predicted_log10: Predicted ``log10`` seconds, shape ``(N,)``.
    :param interval_model: Estimator from :func:`measurement_interval_model`.
    :return: Hit rate, interval width and the extrapolation share.
    :rtype: dict
    """
    points = cohort.half_life_seconds.to_numpy(dtype=float)  # (N,)
    ## REMARK: The estimator was fitted on one assay design and a narrow
    ## half-life domain, so extrapolation is enabled but counted and reported
    ## next to every hit rate it influences.
    lower, upper = interval_model.predict(points, allow_extrapolation=True)  # (N,), (N,)
    outside_domain = (points < interval_model.domain_seconds[0]) | (
        points > interval_model.domain_seconds[1]
    )  # (N,)

    predicted = np.asarray(predicted_log10, dtype=float)  # (N,)
    inside = (predicted >= np.log10(lower)) & (predicted <= np.log10(upper))  # (N,)
    width_log10 = np.log10(upper) - np.log10(lower)  # (N,)

    return {
        "peptides": int(points.size),
        "interval_hit_rate": float(inside.mean()),
        "interval_hit_rate_in_domain": float(inside[~outside_domain].mean())
        if (~outside_domain).any()
        else float("nan"),
        "extrapolated_share": float(outside_domain.mean()),
        "median_interval_width_log10": float(np.median(width_log10)),
        "median_interval_factor": float(10.0 ** np.median(width_log10)),
        "interval_family": interval_model.family,
    }


def model_interval_metrics(cohort, bounds_log10):
    """Score a predictor's own predictive interval against the measurements.

    :param pandas.DataFrame cohort: Aggregated cohort with ``log10_half_life``.
    :param numpy.ndarray bounds_log10: Lower and upper bounds, shape ``(N, 2)``.
    :return: Empirical coverage and interval width, or an empty record if absent.
    :rtype: dict
    """
    if bounds_log10 is None:
        return {}
    bounds = np.asarray(bounds_log10, dtype=float)  # (N, 2)
    true = cohort.log10_half_life.to_numpy(dtype=float)  # (N,)
    covered = (true >= bounds[:, 0]) & (true <= bounds[:, 1])  # (N,)
    width = bounds[:, 1] - bounds[:, 0]  # (N,)
    return {
        "model_interval_coverage": float(covered.mean()),
        "model_interval_median_width_log10": float(np.median(width)),
        "model_interval_median_factor": float(10.0 ** np.median(width)),
    }


def classification_metrics(cohort, predicted_log10):
    """Score the duration class implied by a regression prediction.

    Classes are assigned with the contract function, so predicted and measured
    classes use one scheme.

    :param pandas.DataFrame cohort: Aggregated cohort with ``duration_class``.
    :param numpy.ndarray predicted_log10: Predicted ``log10`` seconds, shape ``(N,)``.
    :return: Accuracy, per-class recall, ordinal error and trivial baselines.
    :rtype: dict
    """
    predicted_seconds = 10.0 ** np.asarray(predicted_log10, dtype=float)  # (N,)
    predicted_class = pd.Series([point_class(value) for value in predicted_seconds])
    true_class = cohort.duration_class.reset_index(drop=True)

    correct = (predicted_class.to_numpy() == true_class.to_numpy())  # (N,)
    index = {name: position for position, name in enumerate(CLASS_ORDER)}
    true_rank = true_class.map(index).to_numpy(dtype=float)  # (N,)
    predicted_rank = predicted_class.map(index).to_numpy(dtype=float)  # (N,)
    distance = np.abs(predicted_rank - true_rank)  # (N,)

    # Per-class recall and the majority baseline
    recall = {}
    for name in CLASS_ORDER:
        selector = true_class.eq(name).to_numpy()
        recall[f"recall_{name}"] = float(correct[selector].mean()) if selector.any() else float("nan")
    shares = true_class.value_counts(normalize=True)

    return {
        "peptides": int(len(true_class)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.nanmean([recall[f"recall_{name}"] for name in CLASS_ORDER])),
        "majority_class_accuracy": float(shares.max()),
        "adjacent_or_exact_accuracy": float((distance <= 1).mean()),
        "mean_class_distance": float(distance.mean()),
        "predicted_class_entropy_classes": int(predicted_class.nunique()),
        **recall,
        **{
            f"predicted_share_{name}": float((predicted_class == name).mean())
            for name in CLASS_ORDER
        },
    }


def confusion_table(cohort, predicted_log10):
    """Build the duration-class confusion matrix.

    :param pandas.DataFrame cohort: Aggregated cohort with ``duration_class``.
    :param numpy.ndarray predicted_log10: Predicted ``log10`` seconds, shape ``(N,)``.
    :return: Confusion matrix with measured classes as rows.
    :rtype: pandas.DataFrame
    """
    predicted_class = [point_class(value) for value in 10.0 ** np.asarray(predicted_log10, dtype=float)]
    table = pd.crosstab(
        pd.Categorical(cohort.duration_class, categories=CLASS_ORDER),
        pd.Categorical(predicted_class, categories=CLASS_ORDER),
        rownames=["measured"],
        colnames=["predicted"],
        dropna=False,
    )
    return table.reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)


def evaluate_predictor(predictor, cohort, interval_model=None):
    """Score one predictor on one cohort across all three questions.

    :param model_benchmark.predictors.base.HalfLifePredictor predictor: Loaded adapter.
    :param pandas.DataFrame cohort: Aggregated cohort from the dataset module.
    :param interval_model: Optional pre-fitted measurement-interval estimator.
    :return: Predictions, the merged metric record and the confusion matrix.
    :rtype: tuple[numpy.ndarray, dict, pandas.DataFrame]
    """
    sequences = cohort.sequence.tolist()
    predicted = predictor.predict_log10_seconds(sequences)  # (N,)
    if interval_model is None:
        interval_model, _ = measurement_interval_model()

    record = {"predictor": predictor.name}
    record.update(regression_metrics(cohort.log10_half_life.to_numpy(dtype=float), predicted))
    record.update(measurement_interval_metrics(cohort, predicted, interval_model))
    record.update(classification_metrics(cohort, predicted))

    ## Published predictive interval, where the artifact ships one
    bounds = predictor.predict_interval_log10_seconds(sequences)
    record.update(model_interval_metrics(cohort, bounds))

    record["clipped_predictions"] = int(getattr(predictor, "clipped_predictions", 0))
    logger.info(
        "%s: MAE %.3f log10, Spearman %.3f, class accuracy %.3f",
        predictor.name,
        record["mae_log10"],
        record["spearman_rho"],
        record["accuracy"],
    )
    return predicted, record, confusion_table(cohort, predicted)
