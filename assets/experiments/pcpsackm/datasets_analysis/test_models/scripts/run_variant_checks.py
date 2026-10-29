"""Evaluate the published artifacts under their alternative configurations.

Two configurations are not what the authors deploy, but both are defensible
readings of the same artifacts, and the difference decides whether a weak
result comes from the model or from its deployment:

* the XGBoost sequence artifacts with the shipped ``scaler.joblib`` applied,
  which the published inference script never loads;
* the ``log1p`` artifacts under the deployed output convention, which omits
  the inverse transform.

Usage::

    .venv/bin/python scripts/run_variant_checks.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

MODULE_SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(MODULE_SOURCE) not in sys.path:
    sys.path.insert(0, str(MODULE_SOURCE))

from model_benchmark.config import PRIMARY_COHORT  # noqa: E402
from model_benchmark.metrics import measurement_interval_model  # noqa: E402
from model_benchmark.pipeline import evaluate_model  # noqa: E402

logger = logging.getLogger(__name__)

VARIANT_RUNS = (
    ("peptiverse_xgb_esm_log", {"apply_scaler": True}, "peptiverse_xgb_esm_log__scaler"),
    ("peptiverse_xgb_esm_raw", {"apply_scaler": True}, "peptiverse_xgb_esm_raw__scaler"),
    (
        "peptiverse_transformer_esm_log",
        {"output_convention": "published_inference"},
        "peptiverse_transformer_esm_log__deployed",
    ),
    (
        "peptiverse_xgb_esm_log",
        {"output_convention": "published_inference"},
        "peptiverse_xgb_esm_log__deployed",
    ),
)


def main():
    """Evaluate every alternative configuration.

    :rtype: None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=PRIMARY_COHORT, help="Cohort key to score")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    interval_model, _ = measurement_interval_model()
    for key, options, result_name in VARIANT_RUNS:
        logger.info("Variant run %s with %s on cohort %s", key, options, arguments.cohort)
        evaluate_model(
            key,
            cohort_name=arguments.cohort,
            interval_model=interval_model,
            run_cost=False,
            run_sensitivity=False,
            predictor_options=options,
            result_name=result_name,
        )


if __name__ == "__main__":
    main()
