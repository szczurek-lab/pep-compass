"""Tests of the web-service adapters, without sending any request.

The result page of a real submission is stored as a fixture, so the parsing,
the sequence cross-check, the cache and the rate limiting are tested offline.
No test in this module contacts an operator's server.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from model_benchmark.predictors import webservice
from model_benchmark.predictors.webservice import (
    BLOCKED_SERVICES,
    PlifePredWebService,
    WebServiceError,
)

FIXTURE = Path(__file__).parent / "fixtures/plifepred_result_page.html"
## REMARK: The three peptides and values of the stored result page, in
## submission order.
FIXTURE_SEQUENCES = ["GLFDIIKKIAESF", "YGGFL", "KLAKLAKKLAKLAK"]
FIXTURE_SECONDS = [948.21, 450.11, 833.11]


@pytest.fixture
def offline_service(tmp_path, monkeypatch):
    """Provide a PlifePred adapter that replays the stored result page.

    :param tmp_path: Pytest temporary directory for the cache.
    :param monkeypatch: Pytest monkeypatch fixture.
    :return: Adapter whose transport is replaced by the fixture.
    :rtype: PlifePredWebService
    """
    monkeypatch.setattr(webservice, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        webservice, "_post_multipart", lambda *args, **kwargs: "multi_submitfreq.php?ran=12345"
    )
    monkeypatch.setattr(
        webservice, "_get", lambda *args, **kwargs: FIXTURE.read_text(encoding="utf-8")
    )
    ## REMARK: Zero delay only because no request leaves the process here; the
    ## production default honours the operator's crawl delay.
    return PlifePredWebService(batch_size=8, crawl_delay_seconds=0.0)


def test_result_page_is_parsed_into_seconds(offline_service):
    """Predictions are read from the result table in seconds."""
    seconds = offline_service.predict_seconds(FIXTURE_SEQUENCES)  # (3,)
    np.testing.assert_allclose(seconds, FIXTURE_SECONDS, rtol=1e-6)


def test_predictions_are_converted_to_log10_seconds(offline_service):
    """The common interface reports the same values on the logarithmic scale."""
    logarithmic = offline_service.predict_log10_seconds(FIXTURE_SEQUENCES)  # (3,)
    np.testing.assert_allclose(logarithmic, np.log10(FIXTURE_SECONDS), rtol=1e-9)


def test_sequence_mismatch_is_rejected(offline_service):
    """A table that does not echo the submitted sequence is an error."""
    ## REMARK: The server labels rows by submission order. If that order ever
    ## shifted, predictions would be silently attached to the wrong peptide, so
    ## the echoed sequence must disagree loudly instead.
    with pytest.raises(WebServiceError):
        offline_service.predict_seconds(["GLFDIIKKIAESF", "WWWWW", "KLAKLAKKLAKLAK"])


def test_cache_prevents_further_requests(offline_service, monkeypatch):
    """A repeated query is served from the cache without contacting the server."""
    first = offline_service.predict_seconds(FIXTURE_SEQUENCES)  # (3,)

    def refuse(*args, **kwargs):
        raise AssertionError("a cached query must not contact the service")

    monkeypatch.setattr(webservice, "_post_multipart", refuse)
    monkeypatch.setattr(webservice, "_get", refuse)
    second = offline_service.predict_seconds(FIXTURE_SEQUENCES)  # (3,)
    np.testing.assert_allclose(first, second, rtol=1e-12)


def test_cache_is_persisted_to_disk(offline_service, tmp_path):
    """Stored predictions survive for a later run of the module."""
    offline_service.predict_seconds(FIXTURE_SEQUENCES)
    stored = json.loads((tmp_path / f"{offline_service.cache_name}.json").read_text(encoding="utf-8"))
    assert set(stored) == set(FIXTURE_SEQUENCES)
    assert stored["YGGFL"] == pytest.approx(450.11)


def test_request_count_is_tracked(offline_service):
    """Each batch costs one submission and one result fetch."""
    offline_service.predict_seconds(FIXTURE_SEQUENCES)
    assert offline_service.requests_sent == 2


def test_default_crawl_delay_matches_the_operator_policy():
    """The default delay is the one the operator's robots.txt requests."""
    assert webservice.WebServicePredictor.crawl_delay_seconds == 20.0


def test_unknown_method_is_rejected():
    """Only the two methods the submission form offers are accepted."""
    with pytest.raises(ValueError):
        PlifePredWebService(method="random_forest")


def test_metadata_declares_provenance_and_training_source():
    """The adapter records that it is queried over HTTP and whose data it saw."""
    service = PlifePredWebService(use_cache=False)
    record = service.describe()
    assert record["queried_over_http"] is True
    assert record["training_source"] == "plifepred"
    assert record["native_units"] == "seconds"
    assert "no weights distributed" in record["artifact"]


def test_blocked_services_record_the_observed_evidence():
    """Services that must not be queried carry the reason and the evidence."""
    assert set(BLOCKED_SERVICES) == {"hlp", "pepmsnd"}
    assert "403" in BLOCKED_SERVICES["hlp"]["observed"]
    assert "/cgibin/" in BLOCKED_SERVICES["hlp"]["robots"]
    assert "unreachable" in BLOCKED_SERVICES["pepmsnd"]["decision"]
