"""Registry of published half-life predictors available locally.

Only predictors whose trained artifact can actually be loaded are registered.
Published models that exist solely as a web service are recorded in
:data:`UNAVAILABLE_MODELS`, so the absence of a benchmark number is documented
rather than silently omitted.
"""

from __future__ import annotations

from model_benchmark.predictors.base import HalfLifePredictor, PredictorMetadata
from model_benchmark.predictors.peptiverse import (
    VARIANTS,
    PeptiVersePredictor,
    available_variants,
    verify_batch_equivalence,
)
from model_benchmark.predictors.webservice import (
    BLOCKED_SERVICES,
    PlifePredWebService,
    WebServiceError,
)

## REMARK: Predictors that exist only as a web service are registered
## separately, because querying them costs requests to the operator's server
## and is rate limited by that operator's robots.txt.
WEB_SERVICES = {
    "plifepred_svm": lambda **options: PlifePredWebService(method="svm", **options),
    "plifepred_svm_motif": lambda **options: PlifePredWebService(method="svm_motif", **options),
}

__all__ = [
    "HalfLifePredictor",
    "PredictorMetadata",
    "PeptiVersePredictor",
    "PlifePredWebService",
    "WebServiceError",
    "VARIANTS",
    "WEB_SERVICES",
    "BLOCKED_SERVICES",
    "available_variants",
    "verify_batch_equivalence",
    "build_predictor",
    "registry",
    "UNAVAILABLE_MODELS",
]

## REMARK: Published half-life regressors whose trained parameters are not
## distributed. They cannot be evaluated here, and refitting them would measure
## our own fit rather than the published model.
UNAVAILABLE_MODELS = {
    "hlp": {
        "publication": "Sharma et al. 2014, BMC Bioinformatics (HLP)",
        "target": "half-life in an intestine-like environment",
        "access": "web server at webs.iiitd.edu.in/raghava/hlp",
        "reason": (
            "The operator blocks automated submission (HTTP 403) and its robots.txt "
            "disallows /cgibin/ for all agents; the matrix is also intestinal, not serum"
        ),
    },
    "tan_2024": {
        "publication": "Tan et al. 2024, Briefings in Bioinformatics",
        "target": "half-life across species and organs, with enzymatic cleavage features",
        "access": "no code or weight release located",
        "reason": "Trained artifact not published; the reported R^2 cannot be reproduced",
    },
    "pepmsnd": {
        "publication": "Hu et al. 2025, Digital Discovery (PepMSND)",
        "target": "blood stability as a threshold label, not a half-life regression",
        "access": "web server at model.highslab.com/pepmsnd",
        "reason": (
            "The host did not respond over HTTP or HTTPS during this study; the target is "
            "also a threshold label, and its 635 training records are a source of our dataset"
        ),
    },
    "cavaco_2021": {
        "publication": "Cavaco et al. 2021, Clinical and Translational Science",
        "target": "serum half-life from physicochemical properties",
        "access": "publisher page returned HTTP 403 during this study",
        "reason": "Published coefficients not retrieved, so the fitted model could not be reconstructed",
    },
}


def registry(include_web_services=False):
    """List the predictor keys that can be evaluated.

    :param bool include_web_services: Also list predictors queried over HTTP.
        They are excluded by default, so that a plain run sends no external
        requests.
    :return: Predictor keys in evaluation order.
    :rtype: list[str]
    """
    keys = available_variants()
    if include_web_services:
        keys = keys + sorted(WEB_SERVICES)
    return keys


def build_predictor(key, **options):
    """Instantiate one registered predictor.

    :param str key: Predictor key from :func:`registry`.
    :param options: Adapter options, such as ``device`` or ``batch_size``.
    :return: Loaded predictor behind the batch interface.
    :rtype: HalfLifePredictor
    :raises KeyError: If the key is not registered.
    """
    if key in VARIANTS:
        return PeptiVersePredictor(key, **options)
    if key in WEB_SERVICES:
        ## REMARK: Web-service adapters take no device; a caller passing the
        ## common options should not have to special-case them.
        options.pop("device", None)
        return WEB_SERVICES[key](**options)
    raise KeyError(
        f"Unknown predictor {key!r}; registered: {registry(include_web_services=True)}; "
        f"documented as unavailable: {sorted(UNAVAILABLE_MODELS)}"
    )
