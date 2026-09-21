#!/usr/bin/env python3
"""Map fixed MEROPS serum-protease identifiers to reviewed human UniProt entries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from _common import DEFAULT_TIMEOUT_SECONDS, acquisition_root, raw_file_record, utc_now, write_bytes, write_receipt
from build_serum_manifest import MANIFEST_COLUMNS


UNIPROT_BASE_URL = "https://rest.uniprot.org"
HUMAN_TAXON_ID = 9606


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read and validate the serum-protease query manifest.

    :param path: Manifest TSV path.
    :return: Manifest rows.
    :raises ValueError: If required columns are absent.
    """

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or set(MANIFEST_COLUMNS) - set(reader.fieldnames):
            raise ValueError("Manifest does not contain the required serum-protease columns.")
        return list(reader)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write updated query-manifest rows to TSV.

    :param path: Manifest TSV path.
    :param rows: Updated rows.
    """

    temporary_path = path.with_name(path.name + ".part")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def request_json(
    url: str,
    timeout_seconds: int,
    method: str = "GET",
    form: dict[str, str] | None = None,
    raw_destination: Path | None = None,
) -> dict[str, Any]:
    """Execute one UniProt JSON request.

    :param url: API endpoint URL.
    :param timeout_seconds: Network timeout in seconds.
    :param method: HTTP method.
    :param form: Form fields for POST requests.
    :param raw_destination: Optional path for the unmodified response bytes.
    :return: Decoded JSON object.
    :raises ValueError: If the response is not a JSON object.
    """

    body = urlencode(form).encode("utf-8") if form is not None else None
    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded" if body is not None else "application/json",
            "User-Agent": "pep-compass-serum-acquisition/1.0",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        raw_payload = response.read()
    if raw_destination is not None:
        write_bytes(raw_destination, raw_payload)
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise ValueError("UniProt API did not return a JSON object.")
    return payload


def result_url_with_json_format(url: str) -> str:
    """Request JSON results while preserving a UniProt redirect URL.

    :param url: UniProt mapping result URL.
    :return: Result URL with ``format=json``.
    """

    parsed = urlsplit(url)
    query = dict(item.split("=", 1) if "=" in item else (item, "") for item in parsed.query.split("&") if item)
    query["format"] = "json"
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


def nested_value(payload: dict[str, Any], *keys: str) -> str:
    """Read a nested string value without assuming optional API fields exist.

    :param payload: Source object.
    :param keys: Nested mapping keys.
    :return: String value or an empty string.
    """

    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return value if isinstance(value, str) else ""


def candidate_from_uniprot(source_code: str, entry: dict[str, Any]) -> dict[str, str] | None:
    """Convert one UniProt mapping result to a candidate human protein.

    :param source_code: MEROPS code submitted to UniProt.
    :param entry: Enriched UniProtKB result object.
    :return: Candidate mapping or ``None`` for a non-human entry.
    """

    organism = entry.get("organism")
    if not isinstance(organism, dict) or organism.get("taxonId") != HUMAN_TAXON_ID:
        return None

    genes = entry.get("genes")
    gene_name = ""
    if isinstance(genes, list) and genes and isinstance(genes[0], dict):
        gene_name = nested_value(genes[0], "geneName", "value")

    ec_numbers: list[str] = []
    protein_description = entry.get("proteinDescription")
    if isinstance(protein_description, dict):
        recommended = protein_description.get("recommendedName")
        if isinstance(recommended, dict):
            values = recommended.get("ecNumbers")
            if isinstance(values, list):
                ec_numbers = [value for value in values if isinstance(value, str)]

    # UniProtKB commonly stores enzyme EC numbers in catalytic-activity comments
    # rather than in the recommended protein name.
    comments = entry.get("comments")
    if isinstance(comments, list):
        for comment in comments:
            if not isinstance(comment, dict) or comment.get("commentType") != "CATALYTIC ACTIVITY":
                continue
            ec_number = nested_value(comment, "reaction", "ecNumber")
            if ec_number:
                ec_numbers.append(ec_number)

    ensembl_ids: list[str] = []
    cross_references = entry.get("uniProtKBCrossReferences")
    if isinstance(cross_references, list):
        for reference in cross_references:
            if isinstance(reference, dict) and reference.get("database") == "Ensembl":
                identifier = reference.get("id")
                if isinstance(identifier, str):
                    ensembl_ids.append(identifier)

    accession = entry.get("primaryAccession")
    if not isinstance(accession, str) or not accession:
        return None
    return {
        "merops_code": source_code,
        "uniprot_accession": accession,
        "gene_name": gene_name,
        "ensembl_gene_id": ";".join(sorted(set(ensembl_ids))),
        "ec_number": ";".join(sorted(set(ec_numbers))),
        "reviewed": "true" if entry.get("entryType") == "UniProtKB reviewed (Swiss-Prot)" else "false",
    }


def choose_candidate(candidates: list[dict[str, str]]) -> tuple[dict[str, str] | None, str]:
    """Choose one unambiguous human mapping, preferring reviewed UniProtKB.

    :param candidates: Human candidates for one MEROPS code.
    :return: Selected candidate and mapping-status label.
    """

    reviewed = [candidate for candidate in candidates if candidate["reviewed"] == "true"]
    preferred = reviewed if reviewed else candidates
    unique = {candidate["uniprot_accession"]: candidate for candidate in preferred}
    if not unique:
        return None, "unresolved_no_human_mapping"
    if len(unique) != 1:
        return None, "unresolved_multiple_human_mappings"
    candidate = next(iter(unique.values()))
    return candidate, "mapped_reviewed" if candidate["reviewed"] == "true" else "mapped_unreviewed"


def add_merops_cross_reference_candidates(
    payload: dict[str, Any],
    expected_codes: set[str],
    grouped: dict[str, list[dict[str, str]]],
) -> None:
    """Add candidates returned by a constrained UniProt MEROPS-xref fallback.

    :param payload: UniProtKB search response.
    :param expected_codes: MEROPS codes that were unresolved by ID mapping.
    :param grouped: Candidate mappings indexed by MEROPS code.
    """

    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("UniProt MEROPS cross-reference search contains no result list.")
    for entry in results:
        if not isinstance(entry, dict):
            continue
        cross_references = entry.get("uniProtKBCrossReferences")
        if not isinstance(cross_references, list):
            continue
        for reference in cross_references:
            if not isinstance(reference, dict) or reference.get("database") != "MEROPS":
                continue
            code = reference.get("id")
            if not isinstance(code, str) or code not in expected_codes:
                continue
            candidate = candidate_from_uniprot(code, entry)
            if candidate is not None:
                grouped.setdefault(code, []).append(candidate)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=acquisition_root() / "manifests" / "serum_proteases.tsv",
        help="Existing serum-protease manifest TSV.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=acquisition_root() / "raw" / "uniprot",
        help="Directory for raw UniProt mapping responses.",
    )
    parser.add_argument("--base-url", default=UNIPROT_BASE_URL, help=argparse.SUPPRESS)
    parser.add_argument("--poll-seconds", type=float, default=3.0, help="Seconds between mapping-status polls.")
    parser.add_argument("--max-polls", type=int, default=40, help="Maximum mapping-status polls.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help=argparse.SUPPRESS)
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Map all fixed MEROPS IDs and update the manifest.

    :param arguments: Command-line arguments excluding program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)
    if args.poll_seconds <= 0 or args.max_polls < 1:
        print("error: polling configuration must be positive.", file=sys.stderr)
        return 2
    try:
        rows = read_manifest(args.manifest)
    except (OSError, ValueError) as error:
        print(f"error: cannot read serum manifest: {error}", file=sys.stderr)
        return 2

    codes = [row["merops_code"] for row in rows]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    fields_url = f"{args.base_url.rstrip('/')}/configure/idmapping/fields"
    try:
        fields_payload = request_json(
            fields_url,
            args.timeout_seconds,
            raw_destination=args.raw_dir / "idmapping_fields.json",
        )

        submission = request_json(
            f"{args.base_url.rstrip('/')}/idmapping/run",
            args.timeout_seconds,
            method="POST",
            form={"from": "MEROPS", "to": "UniProtKB-Swiss-Prot", "ids": ",".join(codes)},
            raw_destination=args.raw_dir / "mapping_submission.json",
        )
        job_id = submission.get("jobId")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("UniProt mapping submission returned no jobId.")

        status_payload: dict[str, Any] | None = None
        for _ in range(args.max_polls):
            status_payload = request_json(
                f"{args.base_url.rstrip('/')}/idmapping/status/{job_id}",
                args.timeout_seconds,
                raw_destination=args.raw_dir / "mapping_status.json",
            )
            state = status_payload.get("jobStatus")
            if state in {"NEW", "RUNNING"}:
                time.sleep(args.poll_seconds)
                continue
            break
        if status_payload is None or status_payload.get("jobStatus") in {"NEW", "RUNNING"}:
            raise TimeoutError("UniProt mapping job did not finish before the polling limit.")
        details = request_json(
            f"{args.base_url.rstrip('/')}/idmapping/details/{job_id}",
            args.timeout_seconds,
            raw_destination=args.raw_dir / "mapping_details.json",
        )
        result_url = details.get("redirectURL")
        if not isinstance(result_url, str) or not result_url:
            raise ValueError("UniProt mapping details returned no redirectURL.")
        result_payload = request_json(
            result_url_with_json_format(result_url),
            args.timeout_seconds,
            raw_destination=args.raw_dir / "mapping_results.json",
        )
    except (HTTPError, URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"error: UniProt mapping failed: {error}", file=sys.stderr)
        return 1

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    results = result_payload.get("results", [])
    if not isinstance(results, list):
        print("error: UniProt mapping results contain no result list.", file=sys.stderr)
        return 1
    for result in results:
        if not isinstance(result, dict):
            continue
        source_code = result.get("from")
        target = result.get("to")
        if not isinstance(source_code, str) or not isinstance(target, dict):
            continue
        candidate = candidate_from_uniprot(source_code, target)
        if candidate is not None:
            grouped[source_code].append(candidate)

    unresolved_after_mapping = {code for code in codes if not grouped[code]}
    fallback_payload: dict[str, Any] = {"results": []}
    if unresolved_after_mapping:
        # The documented ID-mapping endpoint currently omits many otherwise valid
        # MEROPS cross-references. This remains an identifier-based lookup, not a
        # free-text protein-name search, and is executed only for failed mappings.
        terms = " OR ".join(f"xref:MEROPS-{code}" for code in sorted(unresolved_after_mapping))
        fallback_url = (
            f"{args.base_url.rstrip('/')}/uniprotkb/search?"
            f"{urlencode({'query': f'({terms}) AND organism_id:9606 AND reviewed:true', 'format': 'json', 'size': '500'})}"
        )
        try:
            fallback_payload = request_json(
                fallback_url,
                args.timeout_seconds,
                raw_destination=args.raw_dir / "merops_cross_reference_fallback.json",
            )
            add_merops_cross_reference_candidates(fallback_payload, unresolved_after_mapping, grouped)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as error:
            print(f"error: UniProt MEROPS cross-reference fallback failed: {error}", file=sys.stderr)
            return 1

    unresolved: list[dict[str, str]] = []
    for row in rows:
        candidate, status = choose_candidate(grouped[row["merops_code"]])
        row["mapping_status"] = status
        if candidate is None:
            unresolved.append({"merops_code": row["merops_code"], "reason": status})
            continue
        for field in ("uniprot_accession", "gene_name", "ensembl_gene_id", "ec_number"):
            row[field] = candidate[field]

    write_manifest(args.manifest, rows)
    reports_dir = acquisition_root() / "reports"
    unresolved_path = reports_dir / "unresolved_serum_proteases.tsv"
    unresolved_path.parent.mkdir(parents=True, exist_ok=True)
    with unresolved_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("merops_code", "reason"), delimiter="\t")
        writer.writeheader()
        writer.writerows(unresolved)

    missing_ec = [
        {
            "merops_code": row["merops_code"],
            "uniprot_accession": row["uniprot_accession"],
            "mapping_status": row["mapping_status"],
        }
        for row in rows
        if row["include_in_hydrolysis_model"] == "true"
        and row["mapping_status"].startswith("mapped_")
        and not row["ec_number"]
    ]
    missing_ec_path = reports_dir / "mapped_serum_proteases_without_ec.tsv"
    with missing_ec_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("merops_code", "uniprot_accession", "mapping_status"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(missing_ec)

    response_file_names = [
        "idmapping_fields.json",
        "mapping_submission.json",
        "mapping_status.json",
        "mapping_details.json",
        "mapping_results.json",
    ]
    if unresolved_after_mapping:
        response_file_names.append("merops_cross_reference_fallback.json")
    response_files = [
        raw_file_record(args.raw_dir / file_name, fields_url if file_name == "idmapping_fields.json" else "UniProt ID mapping API", "downloaded")
        for file_name in response_file_names
    ]
    write_receipt(
        args.raw_dir / "download_manifest.json",
        "UniProt",
        job_id,
        f"MEROPS -> UniProtKB-Swiss-Prot; ids={','.join(codes)}",
        response_files,
        status="complete" if not unresolved else "completed_with_unresolved_mappings",
    )
    print(
        f"Mapped {len(rows) - len(unresolved)} of {len(rows)} MEROPS entries; "
        f"unresolved: {unresolved_path}; without EC: {missing_ec_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
