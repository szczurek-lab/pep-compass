"""Compare this module's adapter with the authors' own inference script.

The adapter re-implements the published forward path in batched form. This
script runs the upstream ``inference.py`` from the downloaded snapshot on the
same peptides and compares predictions one by one, which is the decisive test
of whether a weak evaluation result describes the published model or an adapter
defect.

The upstream manifest selects a transformer head on ESM-2 for sequence input
and XGBoost on PeptideCLM for SMILES input, so those two artifacts are
compared.

Usage::

    .venv/bin/python scripts/compare_with_upstream.py
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
from model_benchmark.predictors import build_predictor  # noqa: E402
from model_benchmark.predictors.peptiverse import sequence_to_smiles  # noqa: E402

logger = logging.getLogger(__name__)

## REMARK: The upstream manifest maps the half-life property to these two
## artifacts, which fixes the adapter variants to compare against.
UPSTREAM_PAIRS = {
    "wt": "peptiverse_transformer_esm_log",
    "smiles": "peptiverse_xgb_peptideclm",
}


def load_upstream_predictor():
    """Import and construct the authors' predictor from the snapshot.

    :return: Upstream predictor instance.
    :rtype: object
    :raises FileNotFoundError: If the snapshot lacks the inference script.
    """
    script = PEPTIVERSE_DIR / "inference.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing {script}; run scripts/download_peptiverse.py")

    ## REMARK: The upstream script resolves its manifest and tokenizer paths
    ## relative to the working directory, so it is imported with the snapshot
    ## directory on the path and used from there.
    sys.path.insert(0, str(PEPTIVERSE_DIR))
    import inference as upstream

    ## REMARK: The upstream predictor loads every property listed in the
    ## manifest, while only the half-life artifacts were downloaded, so it is
    ## given a manifest restricted to that property.
    manifest = half_life_manifest()
    predictor = upstream.PeptiVersePredictor(
        manifest_path=str(manifest),
        classifier_weight_root=str(PEPTIVERSE_DIR),
    )
    return predictor


def half_life_manifest():
    """Write a manifest containing only the half-life row of the published one.

    :return: Path of the restricted manifest.
    :rtype: pathlib.Path
    """
    source = (PEPTIVERSE_DIR / "basic_models.txt").read_text(encoding="utf-8").splitlines()
    header = source[0]
    rows = [line for line in source[1:] if line.lower().startswith("halflife")]
    target = PEPTIVERSE_DIR / "basic_models_half_life_only.txt"
    target.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    logger.info("Restricted manifest: %s", rows)
    return target


def compare(peptides):
    """Score the same peptides with both implementations.

    :param list[str] peptides: Peptide sequences.
    :return: Per-peptide predictions of both implementations.
    :rtype: pandas.DataFrame
    """
    upstream = load_upstream_predictor()
    records = []

    for column, variant_key in UPSTREAM_PAIRS.items():
        ## REMARK: The comparison runs under the deployed convention, so an
        ## exact match demonstrates that the adapter reproduces the published
        ## forward path. The evaluation itself uses the trained-target
        ## convention, which additionally inverts the log1p transform.
        adapter = build_predictor(variant_key, output_convention="published_inference")
        for sequence in peptides:
            payload = sequence if column == "wt" else sequence_to_smiles(sequence)

            ## Upstream scores one peptide per call
            upstream_result = upstream.predict_property("halflife", column, payload)
            upstream_hours = float(upstream_result["score"])

            ## Adapter scores through the batched path
            adapter_hours = float(adapter.predict_hours([sequence])[0])

            records.append(
                {
                    "column": column,
                    "adapter_variant": variant_key,
                    "sequence": sequence,
                    "upstream_hours": upstream_hours,
                    "adapter_hours": adapter_hours,
                    "absolute_difference_hours": abs(upstream_hours - adapter_hours),
                    "relative_difference": abs(upstream_hours - adapter_hours)
                    / max(abs(upstream_hours), 1e-9),
                }
            )
            logger.info(
                "%s %s: upstream %.6f h, adapter %.6f h",
                column,
                sequence,
                upstream_hours,
                adapter_hours,
            )
        adapter.unload()
    return pd.DataFrame(records)


def main():
    """Run the comparison and store its report.

    :rtype: None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cohort = load_cohort()
    peptides = cohort.sequence.head(6).tolist()

    table = compare(peptides)
    table.to_csv(RESULTS_DIR / "upstream_comparison.csv", index=False)

    summary = {
        "peptides": peptides,
        "max_relative_difference": float(table.relative_difference.max()),
        "max_absolute_difference_hours": float(table.absolute_difference_hours.max()),
        "agrees_within_one_percent": bool((table.relative_difference < 0.01).all()),
        "per_column_max_relative_difference": {
            column: float(group.relative_difference.max())
            for column, group in table.groupby("column")
        },
    }
    (RESULTS_DIR / "upstream_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Adapter agrees with upstream within 1%%: %s", summary["agrees_within_one_percent"])
    logger.info(
        "Largest relative difference: %.2e", summary["max_relative_difference"]
    )


if __name__ == "__main__":
    main()
