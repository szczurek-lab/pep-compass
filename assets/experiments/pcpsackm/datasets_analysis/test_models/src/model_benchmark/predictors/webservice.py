"""Adapters for published half-life predictors available only as a web service.

Some published models distribute no weights and can only be queried through
their authors' server. Where the operator permits automated batch submission,
the model can still be evaluated; where it does not, it must stay unevaluated.

Policy applied here, checked before any request was sent:

* **PlifePred** advertises a batch-submission module, its path is not excluded
  by ``https://webs.iiitd.edu.in/robots.txt``, and the wildcard group of that
  file requests ``Crawl-delay: 20``, which this adapter honours. Peptides are
  sent in batches so that one request covers many sequences.
* **HLP** lives under ``/cgibin/``, which the same ``robots.txt`` disallows, and
  its endpoint answers ``Error 403: Submissions from external sources or
  automated bots are blocked``. It is therefore not queried at all.

Every response is cached on disk, so a repeated evaluation sends no further
requests to the operator.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import numpy as np

from model_benchmark.config import ARTIFACTS_DIR
from model_benchmark.predictors.base import HalfLifePredictor, PredictorMetadata

logger = logging.getLogger(__name__)

CACHE_DIR = ARTIFACTS_DIR / "webservice_cache"
## REMARK: A descriptive agent string, so the operator can attribute and, if
## they wish, block this traffic. Requests are rate limited by robots.txt.
USER_AGENT = "pep-compass-model-benchmark/1.0 (academic evaluation; contact repository owner)"


class WebServiceError(RuntimeError):
    """Raised when a prediction service cannot be queried or parsed."""


def _post_multipart(url, fields, timeout=180):
    """Send a multipart form POST and return the decoded response body.

    :param str url: Endpoint.
    :param dict fields: Form fields as plain strings.
    :param int timeout: Socket timeout in seconds.
    :return: Response body.
    :rtype: str
    :raises WebServiceError: If the request fails or is refused.
    """
    boundary = f"----pepcompass{uuid.uuid4().hex}"
    parts = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n")
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        parts.append(f"{value}\r\n")
    parts.append(f"--{boundary}--\r\n")
    body = "".join(parts).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:200]
        raise WebServiceError(f"{url} returned HTTP {error.code}: {detail}") from error
    except OSError as error:
        raise WebServiceError(f"{url} could not be reached: {error}") from error


def _get(url, timeout=180):
    """Fetch a URL and return the decoded response body.

    :param str url: Address to fetch.
    :param int timeout: Socket timeout in seconds.
    :return: Response body.
    :rtype: str
    :raises WebServiceError: If the request fails.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        raise WebServiceError(f"{url} returned HTTP {error.code}") from error
    except OSError as error:
        raise WebServiceError(f"{url} could not be reached: {error}") from error


class WebServicePredictor(HalfLifePredictor):
    """Half-life predictor queried over HTTP, with an on-disk response cache."""

    #: Seconds to wait between requests, from the operator's robots.txt.
    crawl_delay_seconds = 20.0
    #: Cache file name below ``artifacts/webservice_cache``.
    cache_name = "service"

    def __init__(self, metadata, batch_size=None, crawl_delay_seconds=None, use_cache=True):
        """Initialize the adapter and load its response cache.

        :param PredictorMetadata metadata: Provenance of the queried service.
        :param int batch_size: Peptides per request.
        :param float crawl_delay_seconds: Delay between requests; the service
            default when None.
        :param bool use_cache: Reuse previously stored responses.
        :rtype: None
        """
        super().__init__(metadata, batch_size=batch_size)
        self.use_cache = bool(use_cache)
        if crawl_delay_seconds is not None:
            self.crawl_delay_seconds = float(crawl_delay_seconds)
        self._last_request = 0.0
        self.requests_sent = 0
        self.cache_path = CACHE_DIR / f"{self.cache_name}.json"
        self._cache = {}
        if self.use_cache and self.cache_path.is_file():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            logger.info("Loaded %s cached predictions for %s", len(self._cache), self.name)

    def _respect_crawl_delay(self):
        """Sleep so that consecutive requests honour the operator's crawl delay.

        :rtype: None
        """
        waited = time.monotonic() - self._last_request
        if self._last_request and waited < self.crawl_delay_seconds:
            pause = self.crawl_delay_seconds - waited
            logger.debug("Waiting %.1f s before the next request", pause)
            time.sleep(pause)
        self._last_request = time.monotonic()

    def _store_cache(self):
        """Persist the response cache.

        :rtype: None
        """
        if not self.use_cache:
            return
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2) + "\n", encoding="utf-8")

    def _query(self, sequences):
        """Query the service for peptides that are not cached.

        :param list[str] sequences: Peptide sequences.
        :return: Predicted half-life in seconds per sequence.
        :rtype: dict[str, float]
        """
        raise NotImplementedError

    def _predict_log10_seconds(self, sequences):
        """Return cached or freshly queried predictions for one batch.

        :param list[str] sequences: Peptide sequences of one batch.
        :return: Predictions in ``log10`` seconds, shape ``(B,)``.
        :rtype: numpy.ndarray
        :raises WebServiceError: If the service returns no value for a peptide.
        """
        missing = [sequence for sequence in sequences if sequence not in self._cache]
        if missing:
            fresh = self._query(missing)
            self._cache.update(fresh)
            self._store_cache()

        values = []
        for sequence in sequences:
            if sequence not in self._cache:
                raise WebServiceError(f"{self.name} returned no prediction for {sequence}")
            values.append(float(self._cache[sequence]))
        seconds = np.asarray(values, dtype=float)  # (B,)
        return np.log10(np.clip(seconds, 1e-6, None))  # (B,)


class PlifePredWebService(WebServicePredictor):
    """PlifePred batch submission for natural peptides.

    Reference: Mathur et al., PLoS One 2018,
    https://webs.iiitd.edu.in/raghava/plifepred/
    """

    BASE_URL = "https://webs.iiitd.edu.in/raghava/plifepred"
    SUBMIT_URL = f"{BASE_URL}/multi_pepsearch.php"
    RESULT_URL = f"{BASE_URL}/multi_submitfreq.php"
    default_batch_size = 64
    cache_name = "plifepred"

    ## REMARK: Each result row renders the sequence as a nested table of
    ## coloured single-residue cells, so neither the sequence nor the value can
    ## be taken from a fixed cell position. The sequence is read from the
    ## per-peptide link the row carries, and the prediction from the last cell
    ## before the end of the row.
    ROW_PATTERN = re.compile(
        r"<td[^>]*>\s*pep(?P<index>\d+)\s*</td>.*?"
        r"pepsearch1\.php\?seq=(?P<sequence>[A-Za-z]+).*?"
        r"<td[^>]*>\s*(?P<value>[0-9]+(?:\.[0-9]+)?)\s*</td>\s*</tr>",
        re.IGNORECASE | re.DOTALL,
    )
    JOB_PATTERN = re.compile(r"multi_submitfreq\.php\?ran=(?P<job>\d+)")

    def __init__(self, method="svm", batch_size=None, crawl_delay_seconds=None, use_cache=True):
        """Configure the batch submission.

        :param str method: ``svm`` for the SVM model, ``svm_motif`` for the
            SVM plus motif model offered by the same form.
        :param int batch_size: Peptides per submission.
        :param float crawl_delay_seconds: Delay between requests.
        :param bool use_cache: Reuse previously stored responses.
        :rtype: None
        :raises ValueError: If the method is unknown.
        """
        if method not in {"svm", "svm_motif"}:
            raise ValueError("method must be 'svm' or 'svm_motif'")
        self.method = method
        self.cache_name = f"plifepred_{method}"
        super().__init__(
            PredictorMetadata(
                name=f"plifepred_{method}",
                publication="Mathur et al. 2018, PLoS One (PlifePred)",
                artifact=f"web service {self.SUBMIT_URL} (no weights distributed)",
                representation="amino-acid and dipeptide composition with molecular descriptors",
                architecture="support vector regression" + (" with motif search" if method == "svm_motif" else ""),
                training_records="163 natural peptides (reported)",
                native_units="seconds",
                supports_gradient=False,
                details={
                    "target_matrix": "blood",
                    ## REMARK: Our processed dataset carries a plifepred source,
                    ## which is this model's own training set, so the overlap is
                    ## split off exactly as for the PeptiVerse artifacts.
                    "training_source": "plifepred",
                    "queried_over_http": True,
                    "crawl_delay_seconds": crawl_delay_seconds or self.crawl_delay_seconds,
                },
            ),
            batch_size=batch_size,
            crawl_delay_seconds=crawl_delay_seconds,
            use_cache=use_cache,
        )

    def _submit(self, sequences):
        """Submit one batch and return the job identifier.

        :param list[str] sequences: Peptide sequences.
        :return: Job identifier assigned by the server.
        :rtype: str
        :raises WebServiceError: If the server returns no job identifier.
        """
        fasta = "\n".join(
            f">pep{index}\n{sequence}" for index, sequence in enumerate(sequences)
        )
        self._respect_crawl_delay()
        body = _post_multipart(
            self.SUBMIT_URL,
            {
                "seq": fasta,
                "method": "1" if self.method == "svm" else "2",
                "eval": "10",
                "thval": "0.0",
            },
        )
        self.requests_sent += 1
        match = self.JOB_PATTERN.search(body)
        if not match:
            raise WebServiceError("PlifePred returned no job identifier for this batch")
        return match.group("job")

    def _collect(self, job, expected):
        """Fetch and parse the result table of one job.

        :param str job: Job identifier.
        :param list[str] expected: Sequences submitted in this batch, in order.
        :return: Predicted half-life in seconds per sequence.
        :rtype: dict[str, float]
        :raises WebServiceError: If the result table cannot be parsed.
        """
        self._respect_crawl_delay()
        body = _get(f"{self.RESULT_URL}?ran={job}")
        self.requests_sent += 1

        predictions = {}
        for match in self.ROW_PATTERN.finditer(body):
            index = int(match.group("index"))
            returned = match.group("sequence").upper()
            if index >= len(expected):
                continue
            submitted = expected[index]
            ## REMARK: The identifier carries the submission order and the row
            ## echoes the sequence, so both are cross-checked; a shifted table
            ## is then detected instead of silently mislabelling predictions.
            if returned != submitted:
                raise WebServiceError(
                    f"PlifePred returned {returned} for pep{index}, expected {submitted}"
                )
            predictions[submitted] = float(match.group("value"))

        if not predictions:
            raise WebServiceError(f"PlifePred job {job} produced no parseable rows")
        return predictions

    def _query(self, sequences):
        """Query the service for the given peptides in batches.

        :param list[str] sequences: Peptide sequences not present in the cache.
        :return: Predicted half-life in seconds per sequence.
        :rtype: dict[str, float]
        """
        predictions = {}
        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start : start + self.batch_size]
            job = self._submit(batch)
            logger.info(
                "PlifePred job %s submitted for %s peptides (%s of %s)",
                job,
                len(batch),
                start + len(batch),
                len(sequences),
            )
            received = self._collect(job, batch)
            missing = set(batch) - set(received)
            if missing:
                logger.warning(
                    "PlifePred job %s returned no value for %s peptides", job, len(missing)
                )
            predictions.update(received)
        return predictions


## REMARK: Recorded so that the reason for an absent benchmark stays in the
## results, with the evidence that was actually observed.
BLOCKED_SERVICES = {
    "hlp": {
        "publication": "Sharma et al. 2014, BMC Bioinformatics (HLP)",
        "endpoint": "https://webs.iiitd.edu.in/cgibin/hlp/pep_both.pl",
        "observed": "HTTP 403: Submissions from external sources or automated bots are blocked",
        "robots": "disallowed for all agents: /cgibin/",
        "decision": "not queried; the operator prohibits automated submission",
        "note": "predicts half-life in an intestine-like environment, not serum",
    },
    "pepmsnd": {
        "publication": "Hu et al. 2025, Digital Discovery (PepMSND)",
        "endpoint": "http://model.highslab.com/pepmsnd",
        "observed": "connection failed over HTTP and HTTPS (no response)",
        "robots": "not retrieved; host unreachable",
        "decision": "not queried; service unreachable",
        "note": "classification target, so not a half-life regression benchmark",
    },
}
