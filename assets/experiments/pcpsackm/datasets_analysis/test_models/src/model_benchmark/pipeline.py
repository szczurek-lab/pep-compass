"""Per-model evaluation runs and their persisted results.

One function evaluates one published predictor completely: accuracy on the
cohort, accuracy after removing peptides the model was trained on, numerical
cost, and sensitivity to single substitutions. Results are written below
``results/<predictor>/`` so that the per-model notebooks and the overview
notebook read identical numbers instead of recomputing them.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from model_benchmark import cost as cost_module
from model_benchmark import sensitivity as sensitivity_module
from model_benchmark.config import (
    PRIMARY_COHORT,
    RESULTS_DIR,
    SENSITIVITY_PEPTIDE_COUNT,
    SENSITIVITY_SEED,
)
from model_benchmark.dataset import load_cohort
from model_benchmark.leakage import peptiverse_overlap, split_by_overlap
from model_benchmark.metrics import (
    baseline_metrics,
    evaluate_predictor,
    measurement_interval_model,
    model_interval_metrics,
    regression_metrics,
)
from model_benchmark.predictors import build_predictor

logger = logging.getLogger(__name__)


def result_directory(predictor_name, cohort_name=PRIMARY_COHORT):
    """Return the directory holding one predictor's results for one cohort.

    :param str predictor_name: Predictor or variant-run key.
    :param str cohort_name: Cohort the run was scored on.
    :return: Existing results directory.
    :rtype: pathlib.Path
    """
    path = RESULTS_DIR / cohort_name / predictor_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path, payload):
    """Write a result record as indented JSON.

    :param pathlib.Path path: Target file.
    :param dict payload: Serializable record.
    :rtype: None
    """
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")


def sensitivity_peptides(cohort, count=SENSITIVITY_PEPTIDE_COUNT, seed=SENSITIVITY_SEED, shortest=False):
    """Select the peptides used for the single-substitution analysis.

    :param pandas.DataFrame cohort: Aggregated cohort.
    :param int count: Number of peptides to scan.
    :param int seed: Seed of the random generator.
    :param bool shortest: Take the shortest peptides instead of a random
        sample. A complete scan costs ``19 * L`` predictions, so this keeps the
        request volume low for predictors queried over HTTP.
    :return: Selected peptide sequences.
    :rtype: list[str]
    """
    candidates = cohort.loc[cohort.length.between(5, 25)]
    if candidates.empty:
        candidates = cohort
    if shortest:
        return candidates.nsmallest(min(count, len(candidates)), "length").sequence.tolist()
    sample = candidates.sample(n=min(count, len(candidates)), random_state=seed)
    return sample.sequence.tolist()


def evaluate_model(
    key,
    cohort_name=PRIMARY_COHORT,
    interval_model=None,
    measured_pairs=None,
    run_cost=True,
    run_sensitivity=True,
    device=None,
    batch_size=None,
    predictor_options=None,
    result_name=None,
    sensitivity_count=None,
    sensitivity_shortest=False,
):
    """Evaluate one published predictor and persist every result table.

    :param str key: Predictor key from the registry.
    :param str cohort_name: Cohort to score.
    :param interval_model: Optional pre-fitted measurement-interval estimator.
    :param pandas.DataFrame measured_pairs: Optional measured single-substitution pairs.
    :param bool run_cost: Measure latency and throughput.
    :param bool run_sensitivity: Run the substitution, permutation and gradient tests.
    :param str device: Torch device string passed to the adapter.
    :param int batch_size: Peptides per forward pass.
    :param dict predictor_options: Extra adapter options, such as ``apply_scaler``.
    :param str result_name: Directory name for the results; the predictor key by
        default. A separate name keeps variant runs, such as a run with the
        shipped feature scaler applied, beside the published configuration.
    :param int sensitivity_count: Peptides to scan in the substitution
        analysis; the configured default when None. A predictor queried over
        HTTP needs a small number, because one scan costs ``19 * L`` requests.
    :param bool sensitivity_shortest: Scan the shortest peptides instead of a
        random sample, which also limits the request volume.
    :return: Summary record of this run, also written to the results directory.
    :rtype: dict
    """
    logger.info("Evaluating %s on cohort %s", key, cohort_name)
    cohort = load_cohort(cohort_name)
    if interval_model is None:
        interval_model, _ = measurement_interval_model()
    predictor = build_predictor(
        key, device=device, batch_size=batch_size, **(predictor_options or {})
    )
    output = result_directory(result_name or key, cohort_name)

    # Accuracy on the full cohort
    predicted, record, confusion = evaluate_predictor(predictor, cohort, interval_model)
    record["cohort"] = cohort_name
    predictions = cohort[
        ["sequence_id", "sequence", "length", "half_life_seconds", "log10_half_life", "duration_class", "modification_free", "sources"]
    ].copy()
    predictions["predicted_log10_seconds"] = predicted
    predictions["residual_log10"] = predictions.predicted_log10_seconds - predictions.log10_half_life
    predictions.to_csv(output / "predictions.csv", index=False)
    confusion.to_csv(output / "confusion_matrix.csv")

    # Accuracy after removing peptides present in the published training set
    overlap, overlap_summary = peptiverse_overlap(cohort)
    unseen, seen = split_by_overlap(cohort, overlap)
    record["training_overlap_peptides"] = overlap_summary["overlapping_peptides"]
    record["training_overlap_share"] = overlap_summary["overlap_share"]
    held_out = {}
    if len(unseen) >= 3:
        unseen_predicted = predicted[cohort.sequence_id.isin(unseen.sequence_id).to_numpy()]  # (N_unseen,)
        held_out = regression_metrics(
            unseen.log10_half_life.to_numpy(dtype=float), unseen_predicted
        )
        record.update({f"heldout_{name}": value for name, value in held_out.items()})
    if len(seen) >= 3:
        seen_predicted = predicted[cohort.sequence_id.isin(seen.sequence_id).to_numpy()]  # (N_seen,)
        seen_metrics = regression_metrics(seen.log10_half_life.to_numpy(dtype=float), seen_predicted)
        record.update({f"trainingoverlap_{name}": value for name, value in seen_metrics.items()})

    ## Held-out split by dataset source, for predictors trained on a source of ours
    training_source = predictor.describe().get("training_source")
    if training_source and "sources" in cohort:
        trained_on = cohort.sources.str.contains(training_source, na=False).to_numpy()
        record["source_overlap_peptides"] = int(trained_on.sum())
        record["source_overlap_share"] = float(trained_on.mean())
        record["training_source"] = training_source
        for label, selector in (("sourceheldout", ~trained_on), ("sourceoverlap", trained_on)):
            if selector.sum() >= 3:
                metrics = regression_metrics(
                    cohort.log10_half_life.to_numpy(dtype=float)[selector], predicted[selector]
                )
                record.update({f"{label}_{name}": value for name, value in metrics.items()})

    # Published predictive interval in both spaces
    interval_records = {}
    for space in ("target", "output"):
        bounds = predictor.predict_interval_log10_seconds(cohort.sequence.tolist(), space=space)
        if bounds is None:
            continue
        interval_records[space] = model_interval_metrics(cohort, bounds)
        interval_records[space]["space"] = space
    if interval_records:
        _write_json(output / "model_intervals.json", interval_records)
        record.update({f"interval_{space}_{name}": value
                       for space, values in interval_records.items()
                       for name, value in values.items() if name != "space"})

    # Numerical cost
    if run_cost:
        batch_cost = cost_module.measure_batch_cost(predictor)
        unbatched = cost_module.measure_unbatched_cost(predictor, count=30)
        projection = cost_module.projected_cost(batch_cost, unbatched)
        batch_cost.to_csv(output / "cost_batched.csv", index=False)
        projection.to_csv(output / "cost_projection.csv", index=False)
        _write_json(output / "cost_unbatched.json", unbatched)
        largest = batch_cost.sort_values("batch_size").iloc[-1]
        record["seconds_per_peptide_batched"] = float(largest.seconds_per_peptide)
        record["seconds_per_peptide_single"] = float(unbatched["seconds_per_peptide_median"])
        record["peptides_per_second_batched"] = float(largest.peptides_per_second)
        record["peak_gpu_megabytes"] = float(largest.peak_gpu_megabytes)
        record["batching_speedup"] = float(
            projection.loc[projection.peptides == 1000, "batching_speedup"].iloc[0]
        )

    # Sensitivity to single substitutions
    if run_sensitivity:
        record.update(
            _run_sensitivity(
                predictor,
                cohort,
                output,
                measured_pairs=measured_pairs,
                count=sensitivity_count,
                shortest=sensitivity_shortest,
            )
        )

    ## Accounting for predictors queried over HTTP
    if hasattr(predictor, "requests_sent"):
        record["service_requests_sent"] = int(predictor.requests_sent)
        record["service_crawl_delay_seconds"] = float(predictor.crawl_delay_seconds)

    # Provenance and baselines
    record["model"] = predictor.describe()
    record["cohort_baseline"] = baseline_metrics(cohort.log10_half_life.to_numpy(dtype=float))
    record["measurement_noise_floor_log10"] = float(
        cohort.loc[cohort.record_count > 1, "log10_spread"].median()
        if (cohort.record_count > 1).any()
        else 0.0
    )
    _write_json(output / "summary.json", record)
    predictor.unload()
    logger.info("Wrote results for %s to %s", key, output)
    return record


def _run_sensitivity(predictor, cohort, output, measured_pairs=None, count=None, shortest=False):
    """Run and persist the sensitivity analysis of one predictor.

    :param model_benchmark.predictors.base.HalfLifePredictor predictor: Loaded adapter.
    :param pandas.DataFrame cohort: Aggregated cohort.
    :param pathlib.Path output: Results directory of this predictor.
    :param pandas.DataFrame measured_pairs: Optional measured substitution pairs.
    :param int count: Peptides to scan; the configured default when None.
    :param bool shortest: Scan the shortest peptides instead of a random sample.
    :return: Aggregated sensitivity record for the summary table.
    :rtype: dict
    """
    noise_floor = float(
        cohort.loc[cohort.record_count > 1, "log10_spread"].median()
        if (cohort.record_count > 1).any()
        else 0.0
    )
    peptides = sensitivity_peptides(
        cohort, count=count or SENSITIVITY_PEPTIDE_COUNT, shortest=shortest
    )

    ## Complete single-substitution scans
    scans, summaries, permutations, gradients = [], [], [], []
    for sequence in peptides:
        scan = sensitivity_module.mutational_scan(predictor, sequence)
        scans.append(scan)
        summaries.append(sensitivity_module.scan_summary(scan, noise_floor=noise_floor))
        permutations.append(sensitivity_module.permutation_sensitivity(predictor, sequence))
        gradients.append(sensitivity_module.gradient_agreement(predictor, sequence, scan))

    scan_frame = pd.concat(scans, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    permutation_frame = pd.DataFrame(permutations)
    gradient_frame = pd.DataFrame(gradients)
    scan_frame.to_csv(output / "mutational_scan.csv", index=False)
    summary_frame.to_csv(output / "scan_summary.csv", index=False)
    permutation_frame.to_csv(output / "permutation_test.csv", index=False)
    gradient_frame.to_csv(output / "gradient_agreement.csv", index=False)

    record = {
        "scan_peptides": int(len(summary_frame)),
        "scan_max_absolute_delta_log10": float(summary_frame.max_absolute_delta_log10.max()),
        "scan_median_absolute_delta_log10": float(summary_frame.median_absolute_delta_log10.median()),
        "scan_median_position_variance_share": float(summary_frame.position_variance_share.median()),
        "scan_identical_prediction_share": float(summary_frame.identical_prediction_share.mean()),
        "permutation_median_absolute_delta_log10": float(
            permutation_frame.median_absolute_permutation_delta_log10.median()
        )
        if "median_absolute_permutation_delta_log10" in permutation_frame
        else float("nan"),
    }
    ## REMARK: Substitution changes composition and order, permutation changes
    ## only order. Their ratio shows how much of the response is positional.
    if np.isfinite(record["permutation_median_absolute_delta_log10"]) and record[
        "scan_median_absolute_delta_log10"
    ] > 0:
        record["permutation_to_substitution_ratio"] = (
            record["permutation_median_absolute_delta_log10"]
            / record["scan_median_absolute_delta_log10"]
        )
    if "spearman_gradient_vs_scan" in gradient_frame:
        available = gradient_frame.loc[gradient_frame.available.fillna(False)]
        if not available.empty:
            record["gradient_scan_spearman_median"] = float(
                available.spearman_gradient_vs_scan.median()
            )

    ## Measured parent-variant pairs
    if measured_pairs is not None and not measured_pairs.empty:
        scored, pair_record = sensitivity_module.score_empirical_pairs(
            predictor, measured_pairs, minimum_effect=max(noise_floor, 0.1)
        )
        if not scored.empty:
            scored.to_csv(output / "measured_pairs.csv", index=False)
        record.update({f"pairs_{name}": value for name, value in pair_record.items() if name != "predictor"})
    return record


## REMARK: Variant runs are stored under "<predictor>__<label>" so that a run
## with the shipped scaler applied, or under the deployed output convention,
## stays beside the published configuration without overwriting it.
VARIANT_SEPARATOR = "__"
VARIANT_LABELS = {
    "scaler": "shipped scaler applied",
    "deployed": "deployed output convention",
}


def load_summaries(keys=None, published_only=False, cohort_name=PRIMARY_COHORT):
    """Collect the persisted per-model summaries of one cohort into one table.

    :param list[str] keys: Result directory names; all stored results when None.
    :param bool published_only: Keep only the published configuration of each
        artifact, excluding the alternative-configuration runs.
    :param str cohort_name: Cohort whose results to collect.
    :return: One row per evaluated run, with its configuration named.
    :rtype: pandas.DataFrame
    """
    root = RESULTS_DIR / cohort_name
    if keys is None:
        keys = sorted(path.name for path in root.glob("*") if (path / "summary.json").is_file())
    if published_only:
        keys = [key for key in keys if VARIANT_SEPARATOR not in key]
    records = []
    for key in keys:
        path = root / key / "summary.json"
        if not path.is_file():
            logger.warning("No stored summary for %s", key)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        flattened = {name: value for name, value in payload.items() if not isinstance(value, dict)}
        flattened["run"] = key
        base, _, label = key.partition(VARIANT_SEPARATOR)
        flattened["artifact"] = base
        flattened["configuration"] = VARIANT_LABELS.get(label, label) if label else "as published"
        model = payload.get("model", {})
        flattened["representation"] = model.get("representation")
        flattened["architecture"] = model.get("architecture")
        flattened["target_transform"] = model.get("target_transform")
        flattened["device"] = model.get("device")
        baseline = payload.get("cohort_baseline", {})
        flattened["baseline_mae_log10"] = baseline.get("mae_log10")
        flattened["baseline_rmse_log10"] = baseline.get("rmse_log10")
        records.append(flattened)
    return pd.DataFrame(records)
