"""Generate the evaluation notebooks of this module.

Eight published artifacts are evaluated with one identical protocol, so their
notebooks are generated from one template instead of being maintained eight
times. The setup, overview and sensitivity notebooks are generated here as
well, so that every notebook reads the persisted results through
``model_benchmark.report``.

Usage::

    .venv/bin/python scripts/make_notebooks.py
    .venv/bin/python scripts/make_notebooks.py --execute
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import nbformat

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from model_benchmark.config import EXTENDED_COHORT, PRIMARY_COHORT  # noqa: E402
from model_benchmark.predictors.peptiverse import VARIANTS  # noqa: E402

logger = logging.getLogger(__name__)
NOTEBOOK_DIR = MODULE_ROOT / "notebooks"

PREAMBLE = """import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/model_benchmark").is_dir())
sys.path.insert(0, str(ROOT / "src"))

%matplotlib inline
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
"""


def _markdown(text):
    """Create a markdown cell.

    :param str text: Cell content.
    :return: Markdown cell.
    :rtype: nbformat.NotebookNode
    """
    return nbformat.v4.new_markdown_cell(text.strip())


def _code(text):
    """Create a code cell.

    :param str text: Cell source.
    :return: Code cell.
    :rtype: nbformat.NotebookNode
    """
    return nbformat.v4.new_code_cell(text.strip())


def _notebook(cells):
    """Assemble a notebook from cells.

    :param list cells: Notebook cells.
    :return: Notebook document.
    :rtype: nbformat.NotebookNode
    """
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return notebook


def setup_notebook():
    """Build the notebook describing cohorts, availability and the interval estimator.

    :return: Notebook document.
    :rtype: nbformat.NotebookNode
    """
    cells = [
        _markdown(
            """
# Evaluation setup: which published models exist, and on which peptides we test them

This module tests **pretrained, published** half-life regressors on our own measurements.
No model is trained or refitted here, so every number describes a published artifact as
distributed by its authors.

This notebook fixes the three things the per-model notebooks depend on: the published
models that can actually be obtained, the evaluation cohort, and the measurement interval
estimated around each measured point.
"""
        ),
        _code(PREAMBLE),
        _markdown(
            """
## Which published half-life models can be evaluated

A published accuracy figure can only be checked when the trained parameters are
distributed. The registry separates artifacts we could load from models that exist only
as a web service or only as a paper.
"""
        ),
        _code(
            """
from model_benchmark.predictors import registry, UNAVAILABLE_MODELS, VARIANTS

available = pd.DataFrame(
    [
        {
            "predictor": key,
            "estimator": VARIANTS[key].estimator,
            "representation": VARIANTS[key].backbone,
            "input": VARIANTS[key].input_kind,
            "target": VARIANTS[key].target,
        }
        for key in registry()
    ]
)
display(available.style.set_caption("Published artifacts loaded locally"))

unavailable = pd.DataFrame(UNAVAILABLE_MODELS).T.reset_index(names="model")
display(unavailable.style.set_caption("Published half-life models that could not be evaluated, with the reason"))

## A published model without weights can still be measured when its authors run a
## service and permit automated use. The evidence for each decision is recorded.
from model_benchmark.predictors import BLOCKED_SERVICES, WEB_SERVICES

print("Queried over HTTP:", ", ".join(sorted(WEB_SERVICES)))
display(pd.DataFrame(BLOCKED_SERVICES).T.reset_index(names="service")
        .style.set_caption("Services that were deliberately not queried, with what was observed"))
"""
        ),
        _markdown(
            """
## Evaluation cohort

The cohort rule is the contract rule: an exact half-life relation with a positive point
estimate, restricted to the canonical amino-acid alphabet and to peptides shorter than
50 residues. Repeated measurements of one peptide are collapsed to their geometric mean,
because a sequence-only predictor cannot distinguish repeated conditions.
"""
        ),
        _code(
            """
from model_benchmark.config import COHORTS, PRIMARY_COHORT
from model_benchmark.dataset import cohort_summary, load_cohort

summaries = []
for name in COHORTS:
    cohort = load_cohort(name)
    row = cohort_summary(cohort)
    row.insert(0, "cohort", name)
    row.insert(1, "description", COHORTS[name]["description"])
    summaries.append(row)
display(pd.concat(summaries, ignore_index=True).style.set_caption("Available cohorts"))

cohort = load_cohort(PRIMARY_COHORT)
print(f"Primary cohort: {PRIMARY_COHORT}, {len(cohort)} peptides")
display(
    cohort[["sequence", "length", "half_life_seconds", "duration_class", "record_count", "modification_free", "sources"]]
    .head(12)
    .style.set_caption("First peptides of the primary cohort")
)
"""
        ),
        _markdown(
            """
## Measurement interval around a measured point

The logarithmic relative-width estimator of
[`0_03_estimation_interval_analysis`](../../half_time/0_03_estimation_interval_analysis.ipynb)
turns a measured half-life into lower and upper limits. It is used here as the tolerance
band that a prediction should fall into.

**Its domain is narrow.** It is fitted on the 16 published Cavaco intervals in 50% human
serum, so predictions for peptides outside that half-life range are scored by
extrapolation. Every interval hit rate is therefore reported together with the share of
extrapolated peptides.
"""
        ),
        _code(
            """
from model_benchmark.metrics import measurement_interval_metrics, measurement_interval_model

interval_model, reported = measurement_interval_model()
print(f"Fitted on {len(reported)} reported intervals from: {', '.join(sorted(reported.source.unique()))}")
print(f"Reference t0: {interval_model.reference_seconds / 60:.1f} min")
print(f"Fitted domain: {interval_model.domain_seconds[0] / 60:.1f}-{interval_model.domain_seconds[1] / 60:.1f} min")
print("Coefficients [intercept, slope] per interval side:")
print(np.asarray(interval_model.coefficients))

points = cohort.half_life_seconds.to_numpy(dtype=float)
lower, upper = interval_model.predict(points, allow_extrapolation=True)
band = pd.DataFrame(
    {
        "half_life_seconds": points,
        "lower_seconds": lower,
        "upper_seconds": upper,
        "width_log10": np.log10(upper) - np.log10(lower),
        "width_factor": (upper / lower),
    }
)
display(band.describe().loc[["min", "50%", "max"]].style.set_caption("Estimated interval width over the primary cohort"))

outside = (points < interval_model.domain_seconds[0]) | (points > interval_model.domain_seconds[1])
print(f"Peptides outside the fitted domain: {outside.sum()} of {len(points)} ({100 * outside.mean():.0f}%)")
"""
        ),
        _markdown(
            """
## Overlap with published training data

The PeptiVerse repository distributes its half-life training table as SMILES, so the
overlap with our cohort is determined exactly by comparing canonical structures. Peptides
in that overlap measure the model's training fit, not its transfer, and are reported
separately in every per-model notebook.
"""
        ),
        _code(
            """
from model_benchmark.leakage import peptiverse_overlap

matched, overlap_summary = peptiverse_overlap(cohort)
display(pd.DataFrame([overlap_summary]).style.set_caption("Overlap between the primary cohort and the published training set"))
if not matched.empty:
    display(
        matched[["sequence", "log10_half_life", "training_log10_seconds", "cohort_minus_training_log10", "sources"]]
        .sort_values("cohort_minus_training_log10", key=abs, ascending=False)
        .head(12)
        .style.set_caption("Overlapping peptides: our curated value against the published training target")
    )
"""
        ),
        _markdown(
            """
## Measured single-substitution pairs

The second analysis question needs peptide pairs that differ in exactly one residue and
were measured under the same condition. These pairs are the only empirical check of
whether a predicted change has the right direction.
"""
        ),
        _code(
            """
from model_benchmark.sensitivity import empirical_substitution_pairs

records = load_cohort(PRIMARY_COHORT, aggregate=False)
pairs = empirical_substitution_pairs(records)
print(f"Measured single-substitution pairs in the primary cohort: {len(pairs)}")
if not pairs.empty:
    display(
        pairs[["parent", "variant", "position", "parent_residue", "variant_residue", "measured_delta_log10"]]
        .sort_values("measured_delta_log10", key=abs, ascending=False)
        .style.set_caption("Measured effect of one substitution, in log10 seconds")
    )
"""
        ),
    ]
    return _notebook(cells)


def model_notebook(key, variant=None, cohort=PRIMARY_COHORT):
    """Build the evaluation notebook of one published predictor.

    :param str key: Predictor key.
    :param model_benchmark.predictors.peptiverse.PeptiVerseVariant variant: Variant
        record for a downloaded artifact, or None for a web service.
    :param str cohort: Cohort whose stored results the notebook reads.
    :return: Notebook document.
    :rtype: nbformat.NotebookNode
    """
    is_service = variant is None
    if is_service:
        heading = f"""
# {key}

PlifePred, published by Mathur et al. (PLoS One 2018), which distributes **no trained
parameters**. It is therefore queried through its authors' batch-submission service, in
batches and with the 20-second crawl delay that
`https://webs.iiitd.edu.in/robots.txt` requests. Every response is cached, so re-running
this notebook sends no further requests.

The model predicts half-life **in blood** in seconds, from amino-acid and dipeptide
composition with molecular descriptors, and was trained on 163 natural peptides.

Results are read from `results/{cohort}/{key}/`, written by
`scripts/run_all_models.py --include-web-services`.
"""
    else:
        heading = f"""
# {key}

Published artifact `training_classifiers/half_life/{variant.folder}` of the PeptiVerse
repository, evaluated as distributed. The estimator is **{variant.estimator}** on
**{variant.backbone}** {variant.input_kind} representations, trained on a
**{variant.target}** target.

Results are read from `results/{cohort}/{key}/`, written by `scripts/run_all_models.py`.
"""
    cells = [
        _markdown(heading),
        _code(PREAMBLE),
        _code(
            f"""
from model_benchmark import report
from model_benchmark.metrics import measurement_interval_model

PREDICTOR = "{key}"
COHORT = "{cohort}"
summary = report.load_result(PREDICTOR, "summary.json", COHORT)
predictions = report.load_result(PREDICTOR, "predictions.csv", COHORT)
interval_model, _ = measurement_interval_model()
noise_floor = summary.get("measurement_noise_floor_log10", 0.0)
print(f"Cohort: {{summary['cohort']}}, peptides: {{summary['peptides']}}")
"""
        ),
        _markdown(
            """
## Provenance of the loaded artifact

The record below describes what was actually loaded, including the conformal calibration
shipped with the artifact and the benchmark the authors stored next to it.
"""
        ),
        _code(
            """
model = summary["model"]
display(pd.Series({k: v for k, v in model.items() if not isinstance(v, dict)}).to_frame("value")
        .style.set_caption("Loaded artifact"))

calibration = model.get("conformal_calibration")
if calibration:
    display(pd.Series(calibration).to_frame("value").style.set_caption("Conformal calibration shipped with the artifact"))

benchmark = model.get("published_benchmark") or {}
if benchmark:
    reported = {k: v for k, v in benchmark.items() if not isinstance(v, dict)}
    display(pd.Series(reported).to_frame("value").style.set_caption("Benchmark reported by the authors, on their own data"))
"""
        ),
        _markdown(
            """
## Batch interface
"""
            + (
                """
The service accepts a batch submission, which this adapter uses: one request carries up to
64 peptides. The check below is served from the response cache, so it costs no request.
"""
                if is_service
                else """
The published inference script scores one peptide per call. This adapter scores a batch,
as the oracles of the main library do, so the equivalence of both paths is verified
numerically rather than assumed.
"""
            )
        ),
        _code(
            f"""
from model_benchmark.predictors import build_predictor, verify_batch_equivalence

predictor = build_predictor(PREDICTOR)
probe = predictions.sequence.head(4).tolist()
equivalence = verify_batch_equivalence(predictor, probe)
print(f"Batched equals single-peptide path: {{equivalence['within_tolerance']}}")
print(f"Largest deviation: {{equivalence['max_absolute_deviation']:.2e}} log10 seconds")

matrix = np.array([list(seq.ljust(max(len(s) for s in probe))) for seq in probe])
print("Oracle-style call on a padded token matrix, shape", matrix.shape, "->", predictor(matrix).shape)
"""
        ),
        _markdown(
            """
## Question 1: point regression

Errors are reported on the base-10 logarithmic scale, where 0.3 means a factor of two.
The constant predictor that always answers the cohort median is shown next to the model,
because a model that cannot beat it carries no usable information about our peptides.
"""
        ),
        _code(
            """
regression_keys = ["peptides", "mae_log10", "rmse_log10", "median_absolute_log10", "bias_log10",
                   "r2", "pearson_r", "spearman_rho", "median_error_factor"]
comparison = pd.DataFrame(
    {
        "model": {k: summary.get(k) for k in regression_keys},
        "median baseline": {k: summary["cohort_baseline"].get(k) for k in regression_keys},
    }
)
display(comparison.style.format(precision=3).set_caption("Point regression against the trivial baseline"))
print(f"Measurement noise floor (median within-peptide spread): {noise_floor:.3f} log10")
print(f"Predictions clipped at the positive floor: {summary.get('clipped_predictions', 0)}")
"""
        ),
        _code(
            """
display(report.prediction_scatter(predictions, interval_model, title=f"{PREDICTOR}: predicted against measured"))
plt.close("all")
display(report.residual_histogram(predictions, noise_floor=noise_floor, title=f"{PREDICTOR}: residuals"))
plt.close("all")
"""
        ),
        _markdown(
            """
## Question 2: does the prediction land inside the measurement interval

Two intervals are scored. The first is the interval this project estimates around the
**measured** point with the logarithmic relative-width estimator. The second is the
conformal interval the authors ship with the artifact, evaluated both in the space its
calibration was fitted in and in the space the published inference script applies it to.
"""
        ),
        _code(
            """
interval_keys = ["interval_hit_rate", "interval_hit_rate_in_domain", "extrapolated_share",
                 "median_interval_width_log10", "median_interval_factor"]
display(pd.Series({k: summary.get(k) for k in interval_keys}).to_frame("value")
        .style.format(precision=3).set_caption("Agreement with the estimated measurement interval"))

model_intervals = report.load_result(PREDICTOR, "model_intervals.json", COHORT)
if model_intervals:
    display(pd.DataFrame(model_intervals).T.style.format(precision=3)
            .set_caption("Conformal interval shipped with the artifact: coverage of our measurements"))
else:
    print("This artifact ships no usable conformal calibration.")
"""
        ),
        _markdown(
            """
## Question 3: duration class

The easiest version of the task: does the prediction fall into the right duration class
of the data contract (short below 30 min, medium up to 6 h, long above)? The majority
class baseline is the reference.
"""
        ),
        _code(
            """
class_keys = ["accuracy", "balanced_accuracy", "majority_class_accuracy", "adjacent_or_exact_accuracy",
              "mean_class_distance", "recall_short", "recall_medium", "recall_long",
              "predicted_share_short", "predicted_share_medium", "predicted_share_long",
              "predicted_class_entropy_classes"]
display(pd.Series({k: summary.get(k) for k in class_keys}).to_frame("value")
        .style.format(precision=3).set_caption("Duration-class assignment"))

confusion = report.load_result(PREDICTOR, "confusion_matrix.csv", COHORT)
display(report.confusion_heatmap(confusion, title=f"{PREDICTOR}: duration classes"))
plt.close("all")
"""
        ),
        _markdown(
            """
## Training overlap

Part of our cohort appears in the published training table. Accuracy is therefore split
into the overlapping part, which reports the model's fit to its own training data, and
the held-out part, which is the only honest estimate of transfer to our measurements.
"""
        ),
        _code(
            """
overlap_keys = [k for k in summary if k.startswith(("training_overlap", "heldout_", "trainingoverlap_"))]
display(pd.Series({k: summary[k] for k in overlap_keys}).to_frame("value")
        .style.format(precision=3).set_caption("Overlap with the PeptiVerse training set"))

source_keys = [k for k in summary if k.startswith(("source_overlap", "sourceheldout_", "sourceoverlap_", "training_source"))]
if source_keys:
    display(pd.Series({k: summary[k] for k in source_keys}).to_frame("value")
            .style.format(precision=3)
            .set_caption("Overlap with this model's own training source inside our dataset"))
"""
        ),
        _markdown(
            """
## Numerical cost
"""
            + (
                """
The cost of a web service is not computation but requests to its operator. The synthetic
timing benchmark is therefore not run against it; the figures below are the requests this
evaluation actually needed, at the crawl delay the operator's robots.txt requests.
"""
                if is_service
                else """
Latency of one peptide, throughput of a batch, and the projection to the batch sizes an
optimisation loop needs. The representation backbone dominates the cost, not the
regression head.
"""
            )
        ),
        _code(
            """
service_keys = [k for k in summary if k.startswith("service_")]
if service_keys:
    display(pd.Series({k: summary[k] for k in service_keys}).to_frame("value")
            .style.set_caption("Requests sent to the operator for this evaluation"))
    delay = summary.get("service_crawl_delay_seconds", 20.0)
    sent = summary.get("service_requests_sent", 0)
    peptides = summary["peptides"]
    print(f"{sent} requests for {peptides} peptides, two per batch (submit and collect).")
    print(f"At the {delay:.0f} s crawl delay that is about {sent * delay / 60:.1f} min of wall clock,")
    print(f"or {sent * delay / max(peptides, 1):.1f} s per peptide, dominated entirely by politeness.")
    print(f"A thousand peptides would need about {2 * np.ceil(1000 / 64) * delay / 60:.0f} min at 64 per batch.")
else:
    batch_cost = report.load_result(PREDICTOR, "cost_batched.csv", COHORT)
    unbatched = report.load_result(PREDICTOR, "cost_unbatched.json", COHORT)
    projection = report.load_result(PREDICTOR, "cost_projection.csv", COHORT)
    display(batch_cost.style.format(precision=4).set_caption("Measured cost per batch size"))
    display(pd.Series(unbatched).to_frame("value").style.set_caption("Single-peptide calls, as the published script scores"))
    display(projection.style.format(precision=3).set_caption("Projected wall-clock cost"))
    display(report.cost_curve(batch_cost, title=f"{PREDICTOR}: cost per peptide"))
    plt.close("all")
"""
        ),
        _markdown(
            """
## Sensitivity to a single substitution

Three tests, in increasing strength. A complete substitution scan measures whether the
output moves at all and whether the movement depends on the edited position. A
permutation test keeps the composition and only reorders residues, separating a
positional model from a composition-only one. Measured parent-variant pairs test the
direction of the predicted change against experiment.
"""
        ),
        _code(
            """
scan = report.load_result(PREDICTOR, "mutational_scan.csv", COHORT)
scan_summary = report.load_result(PREDICTOR, "scan_summary.csv", COHORT)
permutation = report.load_result(PREDICTOR, "permutation_test.csv", COHORT)
gradients = report.load_result(PREDICTOR, "gradient_agreement.csv", COHORT)

display(scan_summary.describe().loc[["mean", "50%", "min", "max"]].style.format(precision=4)
        .set_caption("Substitution scan over the selected peptides"))
display(report.scan_heatmap(scan, title=f"{PREDICTOR}: every single substitution"))
plt.close("all")
"""
        ),
        _code(
            """
display(permutation.describe().loc[["mean", "50%", "max"]].style.format(precision=4)
        .set_caption("Reordering the same composition"))
display(report.permutation_comparison(scan_summary, permutation, title=f"{PREDICTOR}: substitution against reordering"))
plt.close("all")

if "available" in gradients and bool(gradients.available.any()):
    display(gradients.style.format(precision=3).set_caption("Input gradients against the finite-difference scan"))
else:
    print("Input gradients are unavailable for this artifact:",
          gradients.reason.iloc[0] if "reason" in gradients else "tree-based estimator")
"""
        ),
        _code(
            """
pairs = report.load_result(PREDICTOR, "measured_pairs.csv", COHORT)
pair_keys = [k for k in summary if k.startswith("pairs_")]
if pair_keys:
    display(pd.Series({k: summary[k] for k in pair_keys}).to_frame("value")
            .style.format(precision=3).set_caption("Agreement with measured single substitutions"))
if pairs is not None and len(pairs):
    display(report.measured_pairs_scatter(pairs, title=f"{PREDICTOR}: measured against predicted change"))
    plt.close("all")
"""
        ),
        _markdown(
            """
## Verdict for this artifact

Each statement below is a threshold comparison on the numbers above, not an
interpretation.
"""
        ),
        _code(
            """
for line in report.verdict_lines(summary):
    print("-", line)
"""
        ),
    ]
    return _notebook(cells)


def overview_notebook():
    """Build the notebook comparing every evaluated artifact.

    :return: Notebook document.
    :rtype: nbformat.NotebookNode
    """
    cells = [
        _markdown(
            """
# Overview: can any published half-life model be used on our data?

This notebook collects the results of every per-model notebook. It answers three
questions for all artifacts at once: how accurate they are on our human-serum cohort,
what they cost per peptide, and whether their accuracy survives the removal of peptides
they were trained on.
"""
        ),
        _code(PREAMBLE),
        _code(
            """
from model_benchmark import report
from model_benchmark.config import PRIMARY_COHORT
from model_benchmark.pipeline import load_summaries

summaries = load_summaries(cohort_name=PRIMARY_COHORT)
published = summaries.loc[summaries.configuration == "as published"]
variants = summaries.loc[summaries.configuration != "as published"]
print(f"Runs: {len(summaries)} ({len(published)} as published, {len(variants)} alternative configurations)")
print(f"Cohort: {summaries.cohort.unique()}")
if not variants.empty:
    print("Alternative configurations:", ", ".join(sorted(variants.configuration.unique())))
"""
        ),
        _markdown(
            """
## Accuracy

MAE is on the base-10 logarithmic scale; the median error factor is its plain reading.
The baseline column is the constant predictor that always answers the cohort median.
"""
        ),
        _code(
            """
accuracy = report.accuracy_table(summaries)
display(accuracy.style.format(precision=3).set_caption("Point regression and duration classes, all runs"))

beats_baseline = accuracy["MAE log10"] < accuracy["MAE median baseline"]
beats_chance = accuracy["Spearman"].abs() > 0.2
print(f"Artifacts beating the median baseline: {int(beats_baseline.sum())} of {len(accuracy)}")
print(f"Artifacts ordering peptides better than chance: {int(beats_chance.sum())} of {len(accuracy)}")
"""
        ),
        _code(
            """
figure, axes = plt.subplots(figsize=(8.4, 4.4))
order = accuracy.sort_values("MAE log10")
positions = np.arange(len(order))
axes.barh(positions, order["MAE log10"], color="#4477AA", label="model")
axes.plot(order["MAE median baseline"], positions, "o", color="#CC3311", label="median baseline")
axes.set_yticks(positions, order["Run"], fontsize=8)
axes.set_xlabel("mean absolute error, log10 seconds")
axes.set_title("Error of every published artifact against the trivial baseline")
axes.legend(fontsize=8)
figure.tight_layout()
display(figure)
plt.close("all")
"""
        ),
        _markdown(
            """
## Effect of training overlap

A published model evaluated on its own training peptides reports its fit. The held-out
columns are the only transfer estimate.
"""
        ),
        _code(
            """
display(report.leakage_table(summaries).style.format(precision=3)
        .set_caption("Accuracy inside and outside the published training set"))
"""
        ),
        _markdown(
            """
## Effect of the two defects found in the distributed artifacts

Two configurations differ from what the authors deploy. Applying the feature scaler that
ships beside the XGBoost artifacts, which their inference script never loads, and
reproducing their output convention, which omits the inverse of the `log1p` target
transform. The comparison below isolates how much each defect costs.
"""
        ),
        _code(
            """
if not variants.empty:
    comparison = summaries[["artifact", "configuration", "mae_log10", "r2", "spearman_rho",
                            "accuracy", "interval_hit_rate"]]
    pivot = comparison.pivot_table(index="artifact", columns="configuration",
                                   values=["mae_log10", "spearman_rho", "accuracy"])
    display(pivot.style.format(precision=3).set_caption("Published configuration against the alternatives"))
    ## The rank correlation is invariant under the log1p convention, since the transform
    ## is monotone; the error and the class assignment are not.
    display(comparison.sort_values(["artifact", "configuration"]).style.format(precision=3)
            .set_caption("All runs, by artifact and configuration"))
else:
    print("No alternative-configuration runs stored; run scripts/run_variant_checks.py")
"""
        ),
        _markdown(
            """
## Numerical cost

Measured on this machine. The batching speed-up is the ratio between the published
per-peptide calling pattern and the batched path of this module.
"""
        ),
        _code(
            """
costs = report.cost_table(published)
display(costs.style.format(precision=4).set_caption("Cost per peptide and projected batch cost"))

figure, axes = plt.subplots(figsize=(8.4, 4.0))
order = costs.sort_values("s / 1000 peptides")
axes.barh(np.arange(len(order)), order["s / 1000 peptides"], color="#EE7733")
axes.set_yticks(np.arange(len(order)), order["Run"], fontsize=8)
axes.set_xlabel("seconds to score 1000 peptides, batched")
axes.set_title("Numerical cost of one optimisation batch")
figure.tight_layout()
display(figure)
plt.close("all")
"""
        ),
        _markdown(
            """
## Sensitivity summary

The full analysis is in the sensitivity notebook; the table below is the comparison
across artifacts.
"""
        ),
        _code(
            """
display(report.sensitivity_table(summaries).style.format(precision=4)
        .set_caption("Response to a single substitution, all artifacts"))
"""
        ),
        _markdown(
            """
## Verdict per artifact

Threshold comparisons on the numbers above, generated from the stored results.
"""
        ),
        _code(
            """
for run in summaries.run:
    stored = report.load_result(run, "summary.json", PRIMARY_COHORT)
    print(f"=== {run} ===")
    for line in report.verdict_lines(stored):
        print("  -", line)
    print()
"""
        ),
    ]
    return _notebook(cells)


def sensitivity_notebook():
    """Build the cross-model notebook on single-substitution sensitivity.

    :return: Notebook document.
    :rtype: nbformat.NotebookNode
    """
    cells = [
        _markdown(
            """
# Does any published model react correctly to a single substitution?

Peptide optimisation changes one residue at a time, so a stability oracle is only
useful if its output reacts to that change, if the reaction depends on which position
changed, and if the direction of the reaction matches measurements.

This notebook compares all artifacts on the three tests described in the per-model
notebooks, and states which of them could drive a search over single substitutions.
"""
        ),
        _code(PREAMBLE),
        _code(
            """
from model_benchmark import report
from model_benchmark.config import PRIMARY_COHORT
from model_benchmark.dataset import load_cohort
from model_benchmark.pipeline import load_summaries
from model_benchmark.sensitivity import empirical_substitution_pairs

## REMARK: Only the published configurations carry a substitution analysis; the
## alternative-configuration runs were scored for accuracy alone.
summaries = load_summaries(published_only=True, cohort_name=PRIMARY_COHORT)
cohort = load_cohort(PRIMARY_COHORT)
noise_floor = float(cohort.loc[cohort.record_count > 1, "log10_spread"].median())
print(f"Artifacts: {len(summaries)}; measurement noise floor: {noise_floor:.3f} log10")
"""
        ),
        _markdown(
            """
## How much does one substitution move the prediction

The scan replaces every residue of a peptide by all 19 alternatives. The size of the
response must be compared with the measurement noise floor: a response far below it
cannot be resolved experimentally, and a response far above the measured effect size is
a different failure.
"""
        ),
        _code(
            """
display(report.sensitivity_table(summaries).style.format(precision=4)
        .set_caption("Single-substitution response of every artifact"))

figure, axes = plt.subplots(figsize=(8.6, 4.4))
order = summaries.sort_values("scan_median_absolute_delta_log10", ascending=False)
positions = np.arange(len(order))
axes.barh(positions, order.scan_median_absolute_delta_log10, color="#4477AA", label="one substitution")
axes.barh(positions, order.permutation_median_absolute_delta_log10, height=0.45, color="#EE7733", label="reordering only")
axes.axvline(noise_floor, color="black", linestyle="--", linewidth=1.0, label="measurement noise floor")
axes.set_yticks(positions, order.predictor, fontsize=8)
axes.set_xlabel("median absolute change, log10 seconds")
axes.set_title("Response to substitution and to reordering")
axes.legend(fontsize=8)
figure.tight_layout()
display(figure)
plt.close("all")
"""
        ),
        _markdown(
            """
## Positional structure

A composition-only representation still moves when a residue is replaced, but the move
barely depends on the position, and reordering the same residues changes nothing. The
positional variance share and the permutation ratio separate the two cases.
"""
        ),
        _code(
            """
for run in summaries.run:
    scan = report.load_result(run, "mutational_scan.csv", PRIMARY_COHORT)
    if scan is None:
        continue
    parent = scan.parent.iloc[0]
    display(report.scan_heatmap(scan, parent=parent, title=f"{run}: {parent}"))
    plt.close("all")
"""
        ),
        _markdown(
            """
## Gradients against finite differences

For the differentiable artifacts the input gradient should rank positions in the same
order as the exhaustive scan. Disagreement means a gradient-driven search would move in
a direction the model's own predictions do not support.
"""
        ),
        _code(
            """
rows = []
for run in summaries.run:
    gradients = report.load_result(run, "gradient_agreement.csv", PRIMARY_COHORT)
    if gradients is None or "available" not in gradients:
        continue
    available = gradients.loc[gradients.available.fillna(False)]
    if available.empty:
        rows.append({"run": run, "available": False,
                     "reason": gradients.reason.iloc[0] if "reason" in gradients else "not differentiable"})
        continue
    rows.append({
        "run": run,
        "available": True,
        "median_spearman_gradient_vs_scan": float(available.spearman_gradient_vs_scan.median()),
        "median_nonzero_gradient_share": float(available.nonzero_gradient_share.median()),
    })
display(pd.DataFrame(rows).style.format(precision=3).set_caption("Input gradients against the exhaustive scan"))
"""
        ),
        _markdown(
            """
## Measured single substitutions

The empirical test. Pairs differ in exactly one residue and were measured under the same
condition, so their measured difference is comparable. The sample is small, which limits
what can be concluded: it can refute a model, not certify one.
"""
        ),
        _code(
            """
records = load_cohort(PRIMARY_COHORT, aggregate=False)
pairs = empirical_substitution_pairs(records)
print(f"Measured pairs available in the primary cohort: {len(pairs)}")
if not pairs.empty:
    display(pairs[["parent", "variant", "position", "parent_residue", "variant_residue", "measured_delta_log10"]]
            .style.format(precision=3).set_caption("Measured effect of one substitution"))

for run in summaries.run:
    scored = report.load_result(run, "measured_pairs.csv", PRIMARY_COHORT)
    if scored is None or not len(scored):
        continue
    display(report.measured_pairs_scatter(scored, title=f"{run}: measured against predicted change"))
    plt.close("all")
"""
        ),
        _markdown(
            """
## What this means for optimisation in pep-compass

The statements below follow from the tables above.
"""
        ),
        _code(
            """
for run in summaries.run:
    stored = report.load_result(run, "summary.json", PRIMARY_COHORT)
    lines = [line for line in report.verdict_lines(stored)
             if "substitution" in line or "Reordering" in line or "direction" in line]
    print(f"=== {run} ===")
    for line in lines:
        print("  -", line)
    print()
"""
        ),
    ]
    return _notebook(cells)


def cohort_notebook():
    """Build the notebook comparing every run across the evaluation cohorts.

    :return: Notebook document.
    :rtype: nbformat.NotebookNode
    """
    cells = [
        _markdown(
            f"""
# Does the conclusion survive a larger cohort?

The per-model notebooks score 128 peptides measured in human serum in vitro. That is a
small sample, so this notebook repeats the comparison on **{EXTENDED_COHORT}**, which
triples the peptide count by accepting serum and plasma of any species, still in vitro.

In vivo records stay excluded everywhere: an in vivo half-life measures pharmacokinetic
clearance, not proteolytic stability, so mixing it in would change the question.

If a model were merely under-powered on the primary cohort, its accuracy should improve
here. If the conclusion is about the model, the numbers will not move.
"""
        ),
        _code(PREAMBLE),
        _code(
            f"""
from model_benchmark import report
from model_benchmark.config import COHORTS
from model_benchmark.dataset import cohort_summary, load_cohort
from model_benchmark.pipeline import load_summaries

COHORT_NAMES = ["{PRIMARY_COHORT}", "{EXTENDED_COHORT}"]

sizes = []
for name in COHORT_NAMES:
    row = cohort_summary(load_cohort(name))
    row.insert(0, "cohort", name)
    row.insert(1, "description", COHORTS[name]["description"])
    sizes.append(row)
display(pd.concat(sizes, ignore_index=True).style.set_caption("The two evaluation cohorts"))

stored = {{}}
for name in COHORT_NAMES:
    frame = load_summaries(cohort_name=name)
    if frame.empty:
        print(f"No stored results for {{name}}")
        continue
    stored[name] = frame
    print(f"{{name}}: {{len(frame)}} stored runs")
"""
        ),
        _markdown(
            """
## Accuracy on both cohorts

Each cohort has its own trivial baseline, because the half-life distribution differs, so
the model error must be read against the baseline of its own cohort.
"""
        ),
        _code(
            """
rows = []
for name, frame in stored.items():
    part = frame.loc[frame.configuration == "as published"][
        ["run", "mae_log10", "baseline_mae_log10", "r2", "spearman_rho", "accuracy",
         "majority_class_accuracy", "interval_hit_rate", "peptides"]
    ].copy()
    part.insert(0, "cohort", name)
    rows.append(part)
combined = pd.concat(rows, ignore_index=True)
display(combined.style.format(precision=3).set_caption("All published configurations, both cohorts"))

for metric in ("spearman_rho", "mae_log10", "accuracy"):
    pivot = combined.pivot_table(index="run", columns="cohort", values=metric)
    display(pivot.style.format(precision=3).set_caption(f"{metric} by cohort"))
"""
        ),
        _code(
            """
figure, axes = plt.subplots(figsize=(8.8, 4.6))
pivot = combined.pivot_table(index="run", columns="cohort", values="spearman_rho").sort_values(
    COHORT_NAMES[0], ascending=True
)
positions = np.arange(len(pivot))
width = 0.38
for offset, (name, colour) in enumerate(zip(COHORT_NAMES, ("#4477AA", "#EE7733"))):
    if name in pivot:
        axes.barh(positions + (offset - 0.5) * width, pivot[name], height=width,
                  color=colour, label=f"{name} (n={int(combined.loc[combined.cohort == name, 'peptides'].max())})")
axes.axvline(0.0, color="black", linewidth=0.8)
axes.set_yticks(positions, pivot.index, fontsize=8)
axes.set_xlabel("Spearman correlation with measured half-life")
axes.set_title("Ranking ability does not improve on the larger cohort")
axes.legend(fontsize=8)
figure.tight_layout()
display(figure)
plt.close("all")
"""
        ),
        _markdown(
            """
## Measured single-substitution pairs

The larger cohort also yields more measured parent-variant pairs, which is the part of the
analysis most limited by sample size.
"""
        ),
        _code(
            """
from model_benchmark.sensitivity import empirical_substitution_pairs

for name in COHORT_NAMES:
    pairs = empirical_substitution_pairs(load_cohort(name, aggregate=False))
    resolvable = pairs.loc[pairs.measured_delta_log10.abs() >= 0.1] if not pairs.empty else pairs
    print(f"{name}: {len(pairs)} pairs, {len(resolvable)} with a measured effect of at least 0.1 log10")

rows = []
for name, frame in stored.items():
    part = frame.loc[frame.configuration == "as published"][
        ["run", "pairs_pairs", "pairs_sign_agreement", "pairs_spearman_delta",
         "pairs_predicted_to_measured_effect_ratio", "scan_median_absolute_delta_log10"]
    ].copy()
    part.insert(0, "cohort", name)
    rows.append(part)
display(pd.concat(rows, ignore_index=True).style.format(precision=3)
        .set_caption("Agreement with measured substitutions, both cohorts"))
"""
        ),
        _markdown(
            """
## Conclusion

Threshold statements over both cohorts.
"""
        ),
        _code(
            """
for name, frame in stored.items():
    published = frame.loc[frame.configuration == "as published"]
    best = published.sort_values("spearman_rho", ascending=False).iloc[0]
    beats_baseline = int((published.mae_log10 < published.baseline_mae_log10).sum())
    beats_chance = int((published.spearman_rho.abs() > 0.2).sum())
    beats_majority = int((published.accuracy > published.majority_class_accuracy).sum())
    print(f"=== {name} ({int(best.peptides)} peptides) ===")
    print(f"  Best model: {best.run}, Spearman {best.spearman_rho:.3f}, MAE {best.mae_log10:.3f} "
          f"against baseline {best.baseline_mae_log10:.3f}")
    print(f"  Runs beating the median baseline in error: {beats_baseline} of {len(published)}")
    print(f"  Runs ordering peptides better than chance: {beats_chance} of {len(published)}")
    print(f"  Runs beating the majority duration class: {beats_majority} of {len(published)}")
    print()
"""
        ),
    ]
    return _notebook(cells)


def refit_notebook():
    """Build the notebook refitting the published families on our cohort.

    :return: Notebook document.
    :rtype: nbformat.NotebookNode
    """
    cells = [
        _markdown(
            """
# Is the architecture wrong, or only the training data?

The published half-life models are small estimators on top of a frozen representation
backbone, fitted on 130 records. A weak result on our cohort therefore has two possible
causes: the architecture cannot represent peptide stability, or its training set does not
cover our measurement condition.

This notebook separates the two. The backbone stays frozen and the **same estimator
families** are refitted on our human-serum cohort under a sequence-similarity split. This
is the only notebook in the module that fits anything, and every number here describes
our own fit, not a published model.

A similarity split is used rather than a random one because the cohort contains analogue
series; a random split over analogues reports an optimistic figure.
"""
        ),
        _code(PREAMBLE),
        _code(
            """
from model_benchmark import refit, report
from model_benchmark.config import PRIMARY_COHORT
from model_benchmark.dataset import load_cohort

## REMARK: Two cohorts are refitted. The primary one answers the question for the
## condition we care about; the wider serum-and-plasma cohort doubles the sample and
## shows whether the limiting factor is the amount of data or the representation.
COHORT_NAMES = [PRIMARY_COHORT, "human_serum_plasma_in_vitro"]

prepared = {}
for name in COHORT_NAMES:
    cohort = load_cohort(name)
    prepared[name] = {
        "cohort": cohort,
        "embeddings": refit.embed_cohort(cohort),
        "targets": cohort.log10_half_life.to_numpy(dtype=float),
        "clusters": refit.similarity_clusters(cohort.sequence.tolist(), threshold=0.6),
    }
    print(f"{name}: {len(cohort)} peptides, {prepared[name]['clusters'].max() + 1} similarity clusters")

cohort = prepared[PRIMARY_COHORT]["cohort"]
embeddings = prepared[PRIMARY_COHORT]["embeddings"]
targets = prepared[PRIMARY_COHORT]["targets"]
clusters = prepared[PRIMARY_COHORT]["clusters"]
"""
        ),
        _markdown(
            """
## Out-of-fold accuracy of the refitted families

Three families are refitted on the frozen ESM-2 embeddings: a ridge regression, gradient
boosting, and a small dense head. Their out-of-fold predictions are scored exactly like
the published artifacts, so the tables are directly comparable.
"""
        ),
        _code(
            """
results = {}
for name, data in prepared.items():
    frame, metrics = refit.cross_validate_refits(data["embeddings"], data["targets"], data["clusters"])
    metrics.insert(0, "cohort", name)
    metrics.insert(1, "cohort_peptides", len(data["cohort"]))
    results[name] = (frame, metrics)
    display(metrics.style.format(precision=3).set_caption(f"Refitted families on {name}, out-of-fold"))

out_of_fold, refit_metrics = results[PRIMARY_COHORT]
combined = pd.concat([metrics for _, metrics in results.values()], ignore_index=True)
display(
    combined.pivot_table(index="estimator", columns="cohort", values="spearman_rho")
    .style.format(precision=3)
    .set_caption("Rank correlation by cohort size: does more data help?")
)
"""
        ),
        _markdown(
            """
## Refit against the published artifact

The published artifact scores every cohort peptide, including the fifth of the cohort it
was trained on. The refit scores only held-out folds. If the refit is clearly better, the
representation and the estimator family are adequate and the published training set is
the limiting factor.
"""
        ),
        _code(
            """
from model_benchmark.pipeline import load_summaries

summaries = load_summaries(cohort_name=PRIMARY_COHORT)
best_published = summaries.sort_values("spearman_rho", ascending=False).iloc[0]
published_summary = report.load_result(best_published.run, "summary.json", PRIMARY_COHORT)
comparison = refit.compare_with_published(refit_metrics, published_summary)
display(comparison.style.format(precision=3).set_caption("Published artifact against refits on our cohort"))
"""
        ),
        _code(
            """
figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
best_refit = refit_metrics.loc[refit_metrics.estimator != "median_baseline"].sort_values("spearman_rho").iloc[-1]
column = f"predicted_{best_refit.estimator}"

axes[0].scatter(out_of_fold.measured_log10, out_of_fold[column], s=28, color="#4477AA", edgecolor="white")
limits = [out_of_fold.measured_log10.min() - 0.3, out_of_fold.measured_log10.max() + 0.3]
axes[0].plot(limits, limits, color="black", linewidth=1.0)
axes[0].set_xlim(limits); axes[0].set_ylim(limits)
axes[0].set_xlabel("measured, log10 seconds")
axes[0].set_ylabel("refit out-of-fold prediction, log10 seconds")
axes[0].set_title(f"refit: {best_refit.estimator} (Spearman {best_refit.spearman_rho:.3f})")

published_predictions = report.load_result(best_published.run, "predictions.csv", PRIMARY_COHORT)
axes[1].scatter(published_predictions.log10_half_life, published_predictions.predicted_log10_seconds,
                s=28, color="#CC3311", edgecolor="white")
axes[1].plot(limits, limits, color="black", linewidth=1.0)
axes[1].set_xlim(limits); axes[1].set_ylim(limits)
axes[1].set_xlabel("measured, log10 seconds")
axes[1].set_ylabel("published prediction, log10 seconds")
axes[1].set_title(f"published: {best_published.run} (Spearman {best_published.spearman_rho:.3f})")
figure.tight_layout()
display(figure)
plt.close("all")
"""
        ),
        _markdown(
            """
## Does a refitted head react to a single substitution

A refit that reproduces half-life but ignores single substitutions is still unusable for
optimisation, so the refitted estimator is probed with the same exhaustive scans.
"""
        ),
        _code(
            """
scans = refit.refit_sensitivity(embeddings, targets, clusters, cohort.sequence.tolist())
display(scans.style.format(precision=4).set_caption("Refitted ridge head: response to every single substitution"))
print(f"Median response over probes: {scans.median_absolute_delta_log10.median():.4f} log10")
print(f"Published artifacts for comparison, median response: "
      f"{summaries.scan_median_absolute_delta_log10.median():.4f} log10")
"""
        ),
        _markdown(
            """
## Conclusion of the refit experiment

Threshold statements on the numbers above.
"""
        ),
        _code(
            """
baseline = refit_metrics.loc[refit_metrics.estimator == "median_baseline"].iloc[0]
best = refit_metrics.loc[refit_metrics.estimator != "median_baseline"].sort_values("spearman_rho").iloc[-1]

print(f"- Best refit on our cohort: {best.estimator}, Spearman {best.spearman_rho:.3f}, MAE {best.mae_log10:.3f} log10.")
print(f"- Median baseline on the same cohort: MAE {baseline.mae_log10:.3f} log10.")
print(f"- Best published artifact: {best_published.run}, Spearman {best_published.spearman_rho:.3f}, "
      f"MAE {best_published.mae_log10:.3f} log10.")
if best.spearman_rho > float(best_published.spearman_rho) + 0.1:
    print("- The frozen representation plus a small head is adequate on our data; the published")
    print("  training set, not the architecture, limits the published artifacts.")
elif best.spearman_rho < 0.3:
    print("- Refitting the published families on our cohort does not produce a usable predictor")
    print("  either, so a pooled sequence embedding is not sufficient for this task.")
else:
    print("- Refitting improves the result only marginally; neither the published artifact nor")
    print("  this architecture is ready for use as a stability oracle.")
"""
        ),
    ]
    return _notebook(cells)


def main():
    """Generate, and optionally execute, every notebook of this module.

    :rtype: None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute the notebooks after writing them")
    parser.add_argument("--timeout", type=int, default=1800, help="Execution timeout per notebook, in seconds")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    # Notebook set
    documents = {"0_01_evaluation_setup.ipynb": setup_notebook()}
    for index, (key, variant) in enumerate(VARIANTS.items(), start=1):
        documents[f"1_{index:02d}_{key}.ipynb"] = model_notebook(key, variant)
    ## REMARK: The web service has no downloadable artifact, so it is appended
    ## after the eight artifacts rather than interleaved with them.
    documents["1_09_plifepred_svm.ipynb"] = model_notebook("plifepred_svm")
    documents["2_01_results_overview.ipynb"] = overview_notebook()
    documents["2_02_substitution_sensitivity.ipynb"] = sensitivity_notebook()
    documents["2_03_cohort_comparison.ipynb"] = cohort_notebook()
    documents["3_01_refit_on_our_cohort.ipynb"] = refit_notebook()

    for name, document in documents.items():
        path = NOTEBOOK_DIR / name
        nbformat.write(document, path)
        logger.info("Wrote %s (%s cells)", path.name, len(document.cells))

    if not arguments.execute:
        return

    # Execution in place, so the committed notebooks carry their outputs
    for name in documents:
        path = NOTEBOOK_DIR / name
        logger.info("Executing %s", path.name)
        completed = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                "--execute", "--inplace", f"--ExecutePreprocessor.timeout={arguments.timeout}",
                str(path),
            ],
            capture_output=True,
            text=True,
            cwd=str(MODULE_ROOT),
        )
        if completed.returncode:
            logger.error("Execution of %s failed:\n%s", path.name, completed.stderr[-4000:])
        else:
            logger.info("Executed %s", path.name)


if __name__ == "__main__":
    main()
