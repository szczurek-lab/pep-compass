"""Evaluate every locally available published half-life predictor.

Usage::

    .venv/bin/python scripts/run_all_models.py
    .venv/bin/python scripts/run_all_models.py --models peptiverse_xgb_esm_log
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
from model_benchmark.dataset import load_cohort  # noqa: E402
from model_benchmark.metrics import measurement_interval_model  # noqa: E402
from model_benchmark.pipeline import evaluate_model  # noqa: E402
from model_benchmark.predictors import WEB_SERVICES, registry  # noqa: E402
from model_benchmark.sensitivity import empirical_substitution_pairs  # noqa: E402

logger = logging.getLogger(__name__)


def main():
    """Run the evaluation for the selected predictors.

    :rtype: None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None, help="Predictor keys; all when omitted")
    parser.add_argument(
        "--include-web-services",
        action="store_true",
        help="Also query predictors that exist only as a web service, honouring the operator's crawl delay",
    )
    parser.add_argument("--cohort", default=PRIMARY_COHORT, help="Cohort key to score")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="Peptides per forward pass")
    parser.add_argument("--skip-cost", action="store_true", help="Skip the timing measurements")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip the substitution analysis")
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    keys = arguments.models or registry(include_web_services=arguments.include_web_services)
    logger.info("Evaluating %s predictors: %s", len(keys), ", ".join(keys))

    # Shared fixtures, fitted once for every predictor
    interval_model, _ = measurement_interval_model()
    records = load_cohort(arguments.cohort, aggregate=False)
    pairs = empirical_substitution_pairs(records)

    failures = {}
    for key in keys:
        ## REMARK: A web service is charged per request to its operator, so the
        ## synthetic cost benchmark is skipped for it and the substitution scan
        ## is reduced to a few short peptides. Its cost is reported from the
        ## requests the cohort run actually needed.
        is_service = key in WEB_SERVICES
        try:
            evaluate_model(
                key,
                cohort_name=arguments.cohort,
                interval_model=interval_model,
                measured_pairs=pairs,
                run_cost=not arguments.skip_cost and not is_service,
                run_sensitivity=not arguments.skip_sensitivity,
                device=arguments.device,
                batch_size=arguments.batch_size,
                sensitivity_count=2 if is_service else None,
                sensitivity_shortest=is_service,
            )
        except Exception as error:  # noqa: BLE001 - one failing artifact must not stop the study
            logger.error("Evaluation of %s failed: %s", key, error, exc_info=True)
            failures[key] = f"{type(error).__name__}: {error}"

    if failures:
        logger.warning("Failed predictors: %s", failures)
    logger.info("Finished %s of %s predictors", len(keys) - len(failures), len(keys))


if __name__ == "__main__":
    main()
