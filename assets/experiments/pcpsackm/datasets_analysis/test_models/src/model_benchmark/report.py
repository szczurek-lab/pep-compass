"""Tables and figures shared by the evaluation notebooks.

Every notebook reads the persisted result files and presents them through this
module, so the per-model notebooks and the overview notebook cannot drift
apart. Duration colors and the readable time axis come from the half-life
analysis module of the data manager.
"""

from __future__ import annotations

import json
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_benchmark.config import CLASS_ORDER, DATA_MANAGER_SOURCE, PRIMARY_COHORT, RESULTS_DIR

logger = logging.getLogger(__name__)

if str(DATA_MANAGER_SOURCE) not in sys.path:
    sys.path.insert(0, str(DATA_MANAGER_SOURCE))

from proteolysis_data_manager.analysis.half_life.visualization import (  # noqa: E402
    DURATION_COLORS,
    readable_time_axis,
)

CANONICAL_RESIDUES = tuple("ACDEFGHIKLMNPQRSTVWY")


def load_result(predictor_name, filename, cohort_name=PRIMARY_COHORT):
    """Read one persisted result file of a predictor run.

    :param str predictor_name: Predictor or variant-run key.
    :param str filename: File name inside the run's result directory.
    :param str cohort_name: Cohort the run was scored on.
    :return: Parsed JSON record or table, or None when the file is absent.
    :rtype: dict | pandas.DataFrame | None
    """
    path = RESULTS_DIR / cohort_name / predictor_name / filename
    if not path.is_file():
        logger.warning("Missing result file %s", path)
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path, index_col=0 if filename == "confusion_matrix.csv" else None)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def accuracy_table(summaries):
    """Build the regression and class accuracy comparison table.

    :param pandas.DataFrame summaries: Output of :func:`model_benchmark.pipeline.load_summaries`.
    :return: Accuracy columns, sorted by rank correlation.
    :rtype: pandas.DataFrame
    """
    columns = {
        "run": "Run",
        "configuration": "Configuration",
        "representation": "Representation",
        "target_transform": "Target",
        "peptides": "Peptides",
        "mae_log10": "MAE log10",
        "baseline_mae_log10": "MAE median baseline",
        "rmse_log10": "RMSE log10",
        "r2": "R2",
        "spearman_rho": "Spearman",
        "median_error_factor": "Median error factor",
        "accuracy": "Class accuracy",
        "majority_class_accuracy": "Majority class accuracy",
        "balanced_accuracy": "Balanced class accuracy",
        "interval_hit_rate": "Within measurement interval",
    }
    available = [name for name in columns if name in summaries.columns]
    table = summaries[available].rename(columns=columns)
    return table.sort_values("Spearman", ascending=False).reset_index(drop=True)


def leakage_table(summaries):
    """Compare accuracy on peptides inside and outside the published training set.

    :param pandas.DataFrame summaries: Output of :func:`model_benchmark.pipeline.load_summaries`.
    :return: Overlap share with held-out and overlapping accuracy.
    :rtype: pandas.DataFrame
    """
    columns = {
        "run": "Run",
        "configuration": "Configuration",
        "training_overlap_peptides": "Peptides in training set",
        "training_overlap_share": "Overlap share",
        "trainingoverlap_mae_log10": "MAE on overlap",
        "trainingoverlap_spearman_rho": "Spearman on overlap",
        "heldout_mae_log10": "MAE held out",
        "heldout_spearman_rho": "Spearman held out",
        "heldout_r2": "R2 held out",
    }
    available = [name for name in columns if name in summaries.columns]
    return summaries[available].rename(columns=columns).reset_index(drop=True)


def cost_table(summaries):
    """Build the numerical cost comparison table.

    :param pandas.DataFrame summaries: Output of :func:`model_benchmark.pipeline.load_summaries`.
    :return: Latency, throughput and projected cost per predictor.
    :rtype: pandas.DataFrame
    """
    columns = {
        "run": "Run",
        "configuration": "Configuration",
        "device": "Device",
        "seconds_per_peptide_single": "s / peptide (one call)",
        "seconds_per_peptide_batched": "s / peptide (batched)",
        "peptides_per_second_batched": "Peptides / s (batched)",
        "batching_speedup": "Batching speed-up",
        "peak_gpu_megabytes": "Peak GPU MB",
    }
    available = [name for name in columns if name in summaries.columns]
    table = summaries[available].rename(columns=columns)
    if "Peptides / s (batched)" in table:
        ## REMARK: One thousand peptides is the batch an optimisation loop
        ## scores per iteration in this project part.
        table["s / 1000 peptides"] = 1000.0 / table["Peptides / s (batched)"]
    return table.sort_values("s / 1000 peptides" if "s / 1000 peptides" in table else "Run").reset_index(drop=True)


def sensitivity_table(summaries):
    """Build the single-substitution sensitivity comparison table.

    :param pandas.DataFrame summaries: Output of :func:`model_benchmark.pipeline.load_summaries`.
    :return: Substitution response, positional structure and measured agreement.
    :rtype: pandas.DataFrame
    """
    columns = {
        "run": "Run",
        "configuration": "Configuration",
        "scan_median_absolute_delta_log10": "Median |delta| substitution",
        "scan_max_absolute_delta_log10": "Max |delta| substitution",
        "permutation_median_absolute_delta_log10": "Median |delta| permutation",
        "permutation_to_substitution_ratio": "Permutation / substitution",
        "scan_median_position_variance_share": "Positional variance share",
        "scan_identical_prediction_share": "Identical predictions",
        "gradient_scan_spearman_median": "Gradient vs scan Spearman",
        "pairs_pairs": "Measured pairs",
        "pairs_sign_agreement": "Sign agreement",
        "pairs_spearman_delta": "Spearman of delta",
        "pairs_predicted_to_measured_effect_ratio": "Predicted / measured effect",
    }
    available = [name for name in columns if name in summaries.columns]
    return summaries[available].rename(columns=columns).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def prediction_scatter(predictions, interval_model=None, title=None):
    """Plot predicted against measured half-life with the measurement interval.

    :param pandas.DataFrame predictions: Stored per-peptide predictions.
    :param interval_model: Optional measurement-interval estimator to draw as a band.
    :param str title: Optional figure title.
    :return: Figure with the scatter and reference lines.
    :rtype: matplotlib.figure.Figure
    """
    figure, axes = plt.subplots(figsize=(6.4, 5.6))
    measured = 10.0 ** predictions.log10_half_life.to_numpy(dtype=float)  # (N,)
    predicted = 10.0 ** predictions.predicted_log10_seconds.to_numpy(dtype=float)  # (N,)

    ## Measurement interval around the measured point
    if interval_model is not None:
        grid = np.geomspace(measured.min(), measured.max(), 100)  # (100,)
        lower, upper = interval_model.predict(grid, allow_extrapolation=True)  # (100,), (100,)
        axes.fill_between(
            grid, lower, upper, color="#999999", alpha=0.25,
            label="estimated measurement interval",
        )

    for name in CLASS_ORDER:
        selector = (predictions.duration_class == name).to_numpy()
        if not selector.any():
            continue
        axes.scatter(
            measured[selector], predicted[selector], s=26, alpha=0.85,
            color=DURATION_COLORS[name], edgecolor="white", linewidth=0.4, label=name,
        )

    limits = [min(measured.min(), predicted.min()) * 0.6, max(measured.max(), predicted.max()) * 1.6]
    axes.plot(limits, limits, color="black", linewidth=1.0, label="perfect prediction")
    axes.set_xlim(limits)
    axes.set_ylim(limits)
    readable_time_axis(axes, axis="x", scale="log")
    readable_time_axis(axes, axis="y", scale="log")
    axes.set_xlabel("measured half-life")
    axes.set_ylabel("predicted half-life")
    axes.set_title(title or "Predicted against measured half-life")
    axes.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    return figure


def residual_histogram(predictions, noise_floor=None, title=None):
    """Plot the distribution of prediction residuals on the logarithmic scale.

    :param pandas.DataFrame predictions: Stored per-peptide predictions.
    :param float noise_floor: Optional measurement noise level to mark.
    :param str title: Optional figure title.
    :return: Figure with the residual histogram.
    :rtype: matplotlib.figure.Figure
    """
    figure, axes = plt.subplots(figsize=(6.4, 3.6))
    residual = predictions.residual_log10.to_numpy(dtype=float)  # (N,)
    axes.hist(residual, bins=24, color="#4477AA", alpha=0.85, edgecolor="white")
    axes.axvline(0.0, color="black", linewidth=1.0, label="no error")
    axes.axvline(float(np.median(residual)), color="#EE7733", linewidth=1.2, label="median residual")
    if noise_floor:
        for sign in (-1, 1):
            axes.axvline(sign * noise_floor, color="#999999", linestyle="--", linewidth=1.0)
    axes.set_xlabel("predicted minus measured, log10 seconds")
    axes.set_ylabel("peptides")
    axes.set_title(title or "Residual distribution")
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure


def confusion_heatmap(confusion, title=None):
    """Plot the duration-class confusion matrix.

    :param pandas.DataFrame confusion: Stored confusion matrix.
    :param str title: Optional figure title.
    :return: Figure with the annotated matrix.
    :rtype: matplotlib.figure.Figure
    """
    figure, axes = plt.subplots(figsize=(4.4, 3.8))
    values = confusion.to_numpy(dtype=float)  # (3, 3)
    image = axes.imshow(values, cmap="Blues")
    axes.set_xticks(range(len(confusion.columns)), confusion.columns)
    axes.set_yticks(range(len(confusion.index)), confusion.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axes.text(
                column, row, int(values[row, column]), ha="center", va="center",
                color="white" if values[row, column] > values.max() / 2 else "black",
            )
    axes.set_xlabel("predicted class")
    axes.set_ylabel("measured class")
    axes.set_title(title or "Duration classes")
    figure.colorbar(image, ax=axes, shrink=0.8, label="peptides")
    figure.tight_layout()
    return figure


def cost_curve(batch_table, title=None):
    """Plot per-peptide cost against batch size.

    :param pandas.DataFrame batch_table: Stored batched cost measurements.
    :param str title: Optional figure title.
    :return: Figure with the cost curve.
    :rtype: matplotlib.figure.Figure
    """
    figure, axes = plt.subplots(figsize=(6.0, 3.6))
    axes.plot(
        batch_table.batch_size, batch_table.seconds_per_peptide,
        marker="o", color="#4477AA",
    )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("peptides per call")
    axes.set_ylabel("seconds per peptide")
    axes.set_title(title or "Cost per peptide against batch size")
    axes.grid(alpha=0.3, which="both")
    figure.tight_layout()
    return figure


def scan_heatmap(scan, parent=None, title=None):
    """Plot the single-substitution response of one peptide.

    :param pandas.DataFrame scan: Stored mutational scan.
    :param str parent: Parent sequence to display; the first one when None.
    :param str title: Optional figure title.
    :return: Figure with residue-by-position changes.
    :rtype: matplotlib.figure.Figure
    """
    parent = parent or scan.parent.iloc[0]
    selected = scan.loc[scan.parent == parent]
    matrix = (
        selected.pivot_table(index="variant_residue", columns="position", values="delta_log10")
        .reindex(index=list(CANONICAL_RESIDUES))
    )  # (20, L)

    figure, axes = plt.subplots(figsize=(max(6.0, 0.42 * matrix.shape[1] + 2), 5.4))
    extreme = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float)))) or 1e-6
    image = axes.imshow(matrix.to_numpy(dtype=float), cmap="RdBu_r", vmin=-extreme, vmax=extreme, aspect="auto")
    axes.set_yticks(range(matrix.shape[0]), matrix.index)
    axes.set_xticks(range(matrix.shape[1]), [f"{position + 1}\n{parent[position]}" for position in matrix.columns])
    axes.set_xlabel("position and wild-type residue")
    axes.set_ylabel("substituted residue")
    axes.set_title(title or f"Predicted change for every substitution: {parent}")
    figure.colorbar(image, ax=axes, shrink=0.85, label="delta log10 seconds")
    figure.tight_layout()
    return figure


def permutation_comparison(scan_summary, permutation_test, title=None):
    """Compare the response to substitution with the response to reordering.

    :param pandas.DataFrame scan_summary: Stored per-peptide scan summaries.
    :param pandas.DataFrame permutation_test: Stored permutation results.
    :param str title: Optional figure title.
    :return: Figure comparing both response sizes per peptide.
    :rtype: matplotlib.figure.Figure
    """
    merged = scan_summary.merge(permutation_test, on=["predictor", "parent"], how="inner")
    figure, axes = plt.subplots(figsize=(6.0, 5.0))
    axes.scatter(
        merged.median_absolute_delta_log10,
        merged.median_absolute_permutation_delta_log10,
        s=34, color="#EE7733", edgecolor="white",
    )
    limit = float(
        max(
            merged.median_absolute_delta_log10.max(),
            merged.median_absolute_permutation_delta_log10.max(),
        )
    ) * 1.15 or 1.0
    axes.plot([0, limit], [0, limit], color="black", linewidth=1.0, label="equal response")
    axes.set_xlim(0, limit)
    axes.set_ylim(0, limit)
    axes.set_xlabel("median |delta| for one substitution, log10 seconds")
    axes.set_ylabel("median |delta| for reordering, log10 seconds")
    axes.set_title(title or "Substitution against reordering of the same composition")
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure


def verdict_lines(summary):
    """Derive the verdict of one predictor from its stored numbers.

    The statements are threshold comparisons, not interpretation: a predictor
    is called usable only if it orders peptides better than chance, beats the
    constant median baseline, and moves when one residue changes.

    :param dict summary: Stored ``summary.json`` record of one predictor.
    :return: Factual verdict statements.
    :rtype: list[str]
    """
    lines = []
    baseline = summary.get("cohort_baseline", {})

    ## Ranking ability
    spearman = summary.get("spearman_rho", float("nan"))
    if not np.isfinite(spearman):
        lines.append("Rank correlation could not be computed.")
    elif abs(spearman) < 0.2:
        lines.append(
            f"Does not order our peptides: Spearman {spearman:.3f}, which is indistinguishable from chance."
        )
    elif spearman < 0.5:
        lines.append(f"Orders our peptides weakly: Spearman {spearman:.3f}.")
    else:
        lines.append(f"Orders our peptides usefully: Spearman {spearman:.3f}.")

    ## Calibration against the trivial baseline
    mae = summary.get("mae_log10", float("nan"))
    baseline_mae = baseline.get("mae_log10", float("nan"))
    if np.isfinite(mae) and np.isfinite(baseline_mae):
        if mae > baseline_mae:
            lines.append(
                f"Worse than answering the cohort median: MAE {mae:.3f} against {baseline_mae:.3f} log10 "
                f"(factor {10 ** mae:.1f} against {10 ** baseline_mae:.1f}), so it is not calibrated for our cohort."
            )
        else:
            lines.append(
                f"Better calibrated than the cohort median: MAE {mae:.3f} against {baseline_mae:.3f} log10."
            )

    ## Class assignment
    accuracy = summary.get("accuracy", float("nan"))
    majority = summary.get("majority_class_accuracy", float("nan"))
    if np.isfinite(accuracy) and np.isfinite(majority):
        comparison = "below" if accuracy < majority else "above"
        lines.append(
            f"Duration class accuracy {accuracy:.3f}, {comparison} the majority-class baseline {majority:.3f}."
        )

    ## Interval agreement
    hit_rate = summary.get("interval_hit_rate")
    if hit_rate is not None and np.isfinite(hit_rate):
        lines.append(
            f"Lands inside the estimated measurement interval for {100 * hit_rate:.0f}% of peptides."
        )

    ## Training overlap
    overlap = summary.get("training_overlap_share")
    if overlap is not None and np.isfinite(overlap):
        held_out = summary.get("heldout_spearman_rho", float("nan"))
        suffix = (
            f"; on the held-out part Spearman is {held_out:.3f}" if np.isfinite(held_out) else ""
        )
        lines.append(
            f"{100 * overlap:.0f}% of the cohort appears in the published training set{suffix}."
        )

    ## Substitution response
    substitution = summary.get("scan_median_absolute_delta_log10")
    if substitution is not None and np.isfinite(substitution):
        if substitution < 0.01:
            lines.append(
                f"Barely reacts to a single substitution: median |delta| {substitution:.4f} log10, "
                "which is far below measurement noise."
            )
        else:
            lines.append(
                f"Reacts to a single substitution with median |delta| {substitution:.3f} log10 "
                f"(factor {10 ** substitution:.2f})."
            )
    ratio = summary.get("permutation_to_substitution_ratio")
    if ratio is not None and np.isfinite(ratio):
        ## REMARK: A shuffle changes many positions at once while a
        ## substitution changes one, so a positional model must move more
        ## under reordering. A ratio near zero is the signature of a
        ## representation that only reads composition.
        if ratio < 0.1:
            reading = "so the representation reacts to composition and almost ignores residue order"
        elif ratio < 1.0:
            reading = "so residue order matters, but less than replacing a residue outright"
        else:
            reading = (
                "so residue order carries at least as much of the response as the identity "
                "of the replaced residue"
            )
        lines.append(f"Reordering produces {ratio:.2f} of the substitution response, {reading}.")

    ## Agreement with measured substitutions
    pairs = summary.get("pairs_pairs")
    if pairs:
        sign = summary.get("pairs_sign_agreement", float("nan"))
        lines.append(
            f"On {int(pairs)} measured single-substitution pairs the predicted direction agrees "
            f"{100 * sign:.0f}% of the time."
        )
        ## REMARK: Direction alone is not enough. A model whose predicted
        ## effects are an order of magnitude smaller than the measured ones
        ## cannot rank candidate substitutions against measurement noise.
        effect_ratio = summary.get("pairs_predicted_to_measured_effect_ratio")
        measured_effect = summary.get("pairs_median_absolute_measured_delta")
        if effect_ratio is not None and np.isfinite(effect_ratio) and measured_effect:
            lines.append(
                f"Predicted substitution effects are {effect_ratio:.2f} times the measured ones "
                f"(measured median {measured_effect:.2f} log10)."
            )
    return lines


def measured_pairs_scatter(pairs, title=None):
    """Plot predicted against measured change for parent-variant pairs.

    :param pandas.DataFrame pairs: Stored measured pairs with predictions.
    :param str title: Optional figure title.
    :return: Figure with the change comparison.
    :rtype: matplotlib.figure.Figure
    """
    figure, axes = plt.subplots(figsize=(5.6, 5.2))
    measured = pairs.measured_delta_log10.to_numpy(dtype=float)  # (P,)
    predicted = pairs.predicted_delta_log10.to_numpy(dtype=float)  # (P,)
    agreement = np.sign(measured) == np.sign(predicted)  # (P,)
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.scatter(
        measured[agreement], predicted[agreement], s=40, color="#228833",
        edgecolor="white", label="direction agrees",
    )
    axes.scatter(
        measured[~agreement], predicted[~agreement], s=40, color="#CC3311",
        edgecolor="white", label="direction disagrees",
    )
    extent = float(max(np.abs(measured).max(), np.abs(predicted).max())) * 1.2 or 1.0
    axes.plot([-extent, extent], [-extent, extent], color="#999999", linestyle="--", linewidth=1.0, label="equal change")
    axes.set_xlim(-extent, extent)
    axes.set_ylim(-extent, extent)
    axes.set_xlabel("measured change, log10 seconds")
    axes.set_ylabel("predicted change, log10 seconds")
    axes.set_title(title or "Single substitutions measured in human serum")
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure
