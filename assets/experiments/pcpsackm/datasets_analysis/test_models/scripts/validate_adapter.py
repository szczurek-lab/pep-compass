"""Verify the batched adapter before trusting any evaluation result.

A weak result on our cohort is only informative if the adapter reproduces the
published forward path. Three checks establish that:

1. **Training-set recovery.** The SMILES artifacts are scored on the authors'
   own training table. A model evaluated on its training data must fit it well,
   so a poor fit here would indicate an adapter defect rather than a model
   defect.
2. **Batch equivalence.** The batched path must agree with single-peptide calls,
   which is the pattern the published script uses.
3. **Scaler sensitivity.** The XGBoost artifacts ship a feature scaler that the
   published script never applies; the effect of applying it is quantified.

Usage::

    .venv/bin/python scripts/validate_adapter.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from model_benchmark.config import PEPTIVERSE_DIR, RESULTS_DIR  # noqa: E402
from model_benchmark.dataset import load_cohort  # noqa: E402
from model_benchmark.metrics import regression_metrics  # noqa: E402
from model_benchmark.predictors import build_predictor, registry, verify_batch_equivalence  # noqa: E402
from model_benchmark.predictors.peptiverse import VARIANTS, SECONDS_PER_HOUR  # noqa: E402

logger = logging.getLogger(__name__)


def training_set_recovery(limit=120):
    """Score the SMILES artifacts on the authors' own training table.

    The adapter converts peptide sequences to SMILES, but the published table
    carries SMILES directly, so the structures are fed to the backbone without
    any conversion of ours.

    :param int limit: Maximum number of training rows to score.
    :return: Fit quality of each SMILES artifact on its own training data.
    :rtype: pandas.DataFrame
    """
    table = pd.read_csv(PEPTIVERSE_DIR / "training_data/half_life_smiles.csv").head(limit)
    target = np.log10(table.half_life_hours.to_numpy(dtype=float) * SECONDS_PER_HOUR)  # (N,)

    records = []
    for key, variant in VARIANTS.items():
        if variant.input_kind != "smiles":
            continue
        predictor = build_predictor(key)

        ## REMARK: The published SMILES bypass our sequence-to-SMILES step, so
        ## this measures the head and the pooling alone.
        raw = []
        smiles = table.SMILES.tolist()
        for start in range(0, len(smiles), predictor.batch_size):
            raw.append(predictor.predict_from_model_inputs(smiles[start : start + predictor.batch_size]))
        hours = np.clip(np.concatenate(raw), 1e-4, None)  # (N,)
        predicted = np.log10(hours * SECONDS_PER_HOUR)  # (N,)

        metrics = regression_metrics(target, predicted)
        metrics.update({"predictor": key, "rows": int(len(table))})
        records.append(metrics)
        logger.info(
            "%s on its own training data: Spearman %.3f, R2 %.3f, MAE %.3f",
            key,
            metrics["spearman_rho"],
            metrics["r2"],
            metrics["mae_log10"],
        )
        predictor.unload()
    return pd.DataFrame(records)


def batch_equivalence(sequences):
    """Check the batched path against single-peptide calls for every artifact.

    :param list[str] sequences: Probe peptides.
    :return: Equivalence record per artifact.
    :rtype: pandas.DataFrame
    """
    records = []
    for key in registry():
        predictor = build_predictor(key)
        result = verify_batch_equivalence(predictor, sequences)
        records.append(
            {
                "predictor": key,
                "max_absolute_deviation_log10": result["max_absolute_deviation"],
                "within_tolerance": result["within_tolerance"],
            }
        )
        logger.info(
            "%s batch equivalence: %s (max deviation %.2e)",
            key,
            result["within_tolerance"],
            result["max_absolute_deviation"],
        )
        predictor.unload()
    return pd.DataFrame(records)


def scaler_effect(sequences):
    """Quantify the effect of the scaler the published script ignores.

    :param list[str] sequences: Probe peptides.
    :return: Predictions with and without the shipped scaler.
    :rtype: pandas.DataFrame
    """
    records = []
    for key, variant in VARIANTS.items():
        if variant.estimator != "xgb" or variant.backbone != "esm":
            continue
        without = build_predictor(key, apply_scaler=False)
        with_scaler = build_predictor(key, apply_scaler=True)
        plain = without.predict_log10_seconds(sequences)  # (N,)
        scaled = with_scaler.predict_log10_seconds(sequences)  # (N,)
        records.append(
            {
                "predictor": key,
                "scaler_shipped": with_scaler.scaler is not None,
                "median_absolute_difference_log10": float(np.median(np.abs(scaled - plain))),
                "max_absolute_difference_log10": float(np.abs(scaled - plain).max()),
                "median_prediction_without_scaler": float(np.median(plain)),
                "median_prediction_with_scaler": float(np.median(scaled)),
            }
        )
        logger.info(
            "%s scaler effect: median difference %.3f log10",
            key,
            records[-1]["median_absolute_difference_log10"],
        )
        without.unload()
        with_scaler.unload()
    return pd.DataFrame(records)


def main():
    """Run every adapter check and store the report.

    :rtype: None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cohort = load_cohort()
    probes = cohort.sequence.head(6).tolist()

    recovery = training_set_recovery()
    equivalence = batch_equivalence(probes)
    scaler = scaler_effect(probes)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    recovery.to_csv(RESULTS_DIR / "adapter_training_recovery.csv", index=False)
    equivalence.to_csv(RESULTS_DIR / "adapter_batch_equivalence.csv", index=False)
    scaler.to_csv(RESULTS_DIR / "adapter_scaler_effect.csv", index=False)
    (RESULTS_DIR / "adapter_validation.json").write_text(
        json.dumps(
            {
                "probe_peptides": probes,
                "all_batch_paths_equivalent": bool(equivalence.within_tolerance.all()),
                "training_recovery_min_spearman": float(recovery.spearman_rho.min())
                if not recovery.empty
                else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote adapter validation reports to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
