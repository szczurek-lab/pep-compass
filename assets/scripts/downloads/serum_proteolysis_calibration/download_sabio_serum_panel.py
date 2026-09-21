#!/usr/bin/env python3
"""Download normalized SABIO-RK SBML only for mapped serum peptidases."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from _common import DEFAULT_TIMEOUT_SECONDS, acquisition_root, raw_file_record, utc_now, write_json, write_receipt
from build_serum_manifest import MANIFEST_COLUMNS


SABIO_BASE_URL = "https://sabiork.h-its.org/sabioRestWebServices"


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read the completed serum-protease manifest.

    :param path: Serum manifest TSV.
    :return: All manifest rows.
    :raises ValueError: If required columns are absent.
    """

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or set(MANIFEST_COLUMNS) - set(reader.fieldnames):
            raise ValueError("Manifest does not contain the required serum-protease columns.")
        return list(reader)


def is_included(row: dict[str, str]) -> bool:
    """Return whether one manifest row is eligible for kinetic acquisition.

    :param row: Serum manifest row.
    :return: ``True`` for included, uniquely mapped rows with EC identifiers.
    """

    ec_numbers = [value for value in row.get("ec_number", "").split(";") if value]
    return (
        row.get("include_in_hydrolysis_model", "").lower() == "true"
        and row.get("mapping_status", "").startswith("mapped_")
        and len(ec_numbers) == 1
    )


def sabio_query(ec_number: str) -> str:
    """Construct the constrained human EC query required by the acquisition contract.

    :param ec_number: One UniProt-derived EC number.
    :return: SABIO-RK query expression.
    """

    return f'Organism:"Homo sapiens" AND ECNumber:{ec_number}'


def is_sbml(payload: bytes) -> bool:
    """Return whether a response contains an SBML root element.

    :param payload: HTTP response body.
    :return: ``True`` only for SBML XML.
    """

    prefix = payload[:16_384].decode("utf-8", errors="replace")
    return re.search(r"<(?:(?:[A-Za-z_][\w.-]*):)?sbml\b", prefix, re.IGNORECASE) is not None


def get_bytes(url: str, timeout_seconds: int) -> bytes:
    """Fetch one SABIO-RK URL.

    :param url: Fully qualified SABIO-RK URL.
    :param timeout_seconds: Network timeout in seconds.
    :return: Raw response bytes.
    """

    request = Request(
        url,
        headers={
            "Accept": "application/sbml+xml, application/xml, text/plain",
            "User-Agent": "pep-compass-serum-acquisition/1.0",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def write_bytes(path: Path, payload: bytes) -> None:
    """Write one raw response atomically.

    :param path: Destination file.
    :param payload: Unmodified response bytes.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".part")
    temporary_path.write_bytes(payload)
    temporary_path.replace(path)


def entry_ids_from_response(payload: bytes) -> list[str]:
    """Parse the text export returned by SABIO's entry-ID endpoint.

    :param payload: Plain-text endpoint response.
    :return: Unique kinetic-law identifiers in response order.
    :raises ValueError: If a non-empty line is not an integer identifier.
    """

    entry_ids: list[str] = []
    for line in payload.decode("utf-8", errors="strict").splitlines():
        value = line.strip()
        if not value:
            continue
        if not value.isdigit():
            raise ValueError(f"SABIO-RK entry-ID response contains invalid value {value!r}.")
        if value not in entry_ids:
            entry_ids.append(value)
    return entry_ids


def record_directory(output_dir: Path, merops_code: str) -> Path:
    """Return the raw-output directory for one MEROPS entry.

    :param output_dir: SABIO raw root.
    :param merops_code: MEROPS identifier.
    :return: Per-protease output directory.
    """

    return output_dir / merops_code


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=acquisition_root() / "manifests" / "serum_proteases.tsv",
        help="Mapped serum-protease manifest TSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=acquisition_root() / "raw" / "sabio",
        help="Directory for per-MEROPS raw SABIO-RK responses.",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N eligible proteases.")
    parser.add_argument("--dry-run", action="store_true", help="Write the query plan without external requests.")
    parser.add_argument("--force", action="store_true", help="Replace existing raw source artifacts.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after individual source failures.")
    parser.add_argument("--base-url", default=SABIO_BASE_URL, help=argparse.SUPPRESS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help=argparse.SUPPRESS)
    return parser


def download_one(
    row: dict[str, str],
    output_dir: Path,
    base_url: str,
    timeout_seconds: int,
    force: bool,
) -> dict[str, Any]:
    """Download the exact SABIO-RK response set for one mapped protease.

    :param row: Eligible serum-manifest row.
    :param output_dir: SABIO raw-output root.
    :param base_url: SABIO REST base URL.
    :param timeout_seconds: Network timeout in seconds.
    :param force: Replace existing source artifacts.
    :return: Per-protease receipt summary.
    :raises OSError: If the external service fails or returns invalid data.
    :raises ValueError: If a returned response is malformed.
    """

    merops_code = row["merops_code"]
    ec_numbers = [value for value in row["ec_number"].split(";") if value]
    if len(ec_numbers) != 1:
        raise ValueError(f"{merops_code} has no unique EC number: {row['ec_number']!r}.")
    query = sabio_query(ec_numbers[0])
    directory = record_directory(output_dir, merops_code)
    directory.mkdir(parents=True, exist_ok=True)
    entry_path = directory / "entry_ids.txt"
    sbml_path = directory / "kinetic_laws.sbml"
    request_path = directory / "request.json"

    entry_url = f"{base_url.rstrip('/')}/searchKineticLaws/entryIDs?{urlencode({'q': query, 'format': 'txt'})}"
    if entry_path.exists() and not force:
        entry_payload = entry_path.read_bytes()
        entry_ids = entry_ids_from_response(entry_payload)
        entry_status = "retained"
    else:
        entry_payload = get_bytes(entry_url, timeout_seconds)
        entry_ids = entry_ids_from_response(entry_payload)
        write_bytes(entry_path, entry_payload)
        entry_status = "downloaded"

    files = [raw_file_record(entry_path, entry_url, entry_status)]
    sbml_url = ""
    if entry_ids:
        sbml_url = f"{base_url.rstrip('/')}/kineticLaws?{urlencode({'kinlawids': ','.join(entry_ids), 'normalized': 'true'})}"
        if sbml_path.exists() and not force:
            sbml_payload = sbml_path.read_bytes()
            if not is_sbml(sbml_payload):
                raise ValueError(f"Existing SABIO-RK output is not SBML: {sbml_path}.")
            sbml_status = "retained"
        else:
            sbml_payload = get_bytes(sbml_url, timeout_seconds)
            if not is_sbml(sbml_payload):
                raise ValueError("SABIO-RK returned a non-SBML response for kinetic-law IDs.")
            write_bytes(sbml_path, sbml_payload)
            sbml_status = "downloaded"
        files.append(raw_file_record(sbml_path, sbml_url, sbml_status))

    request_payload = {
        "merops_code": merops_code,
        "uniprot_accession": row["uniprot_accession"],
        "ec_number": ec_numbers[0],
        "query": query,
        "entry_ids_url": entry_url,
        "entry_ids": entry_ids,
        "kinetic_laws_url": sbml_url or None,
        "retrieved_at": utc_now(),
        "result": "empty" if not entry_ids else "sbml",
    }
    write_json(request_path, request_payload)
    files.append(raw_file_record(request_path, entry_url, "generated_provenance"))
    write_receipt(
        directory / "download_manifest.json",
        "SABIO-RK",
        merops_code,
        query,
        files,
        status="empty" if not entry_ids else "complete",
    )
    return {
        "merops_code": merops_code,
        "result": request_payload["result"],
        "entry_count": len(entry_ids),
        "files": files,
    }


def main(arguments: list[str] | None = None) -> int:
    """Download all eligible, mapped SABIO-RK source responses.

    :param arguments: Command-line arguments excluding program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be positive.", file=sys.stderr)
        return 2
    try:
        rows = read_manifest(args.manifest)
    except (OSError, ValueError) as error:
        print(f"error: cannot read serum manifest: {error}", file=sys.stderr)
        return 2

    eligible = [row for row in rows if is_included(row)]
    skipped = [
        {"merops_code": row["merops_code"], "reason": "excluded_or_unmapped_or_missing_ec"}
        for row in rows
        if not is_included(row)
    ]
    if args.limit is not None:
        eligible = eligible[: args.limit]
    if not eligible:
        print("error: no mapped, included proteases with exactly one EC number are available.", file=sys.stderr)
        return 2

    plan = [
        {
            "merops_code": row["merops_code"],
            "uniprot_accession": row["uniprot_accession"],
            "ec_number": row["ec_number"],
            "query": sabio_query(row["ec_number"]),
        }
        for row in eligible
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        write_json(
            args.output_dir / "serum_panel_download_manifest.json",
            {"created_at": utc_now(), "dry_run": True, "queries": plan, "skipped": skipped},
        )
        for entry in plan:
            print(f"{entry['merops_code']}\t{entry['query']}")
        print(f"Planned {len(plan)} SABIO-RK requests; skipped {len(skipped)} manifest rows.")
        return 0

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for row in eligible:
        print(f"[{row['merops_code']}] {row['ec_number']}")
        try:
            completed.append(download_one(row, args.output_dir, args.base_url, args.timeout_seconds, args.force))
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, ValueError) as error:
            failures.append({"merops_code": row["merops_code"], "error": str(error)})
            print(f"error: {row['merops_code']}: {error}", file=sys.stderr)
            if not args.continue_on_error:
                break

    write_json(
        args.output_dir / "serum_panel_download_manifest.json",
        {
            "created_at": utc_now(),
            "manifest": str(args.manifest),
            "completed": completed,
            "failed": failures,
            "skipped": skipped,
        },
    )
    if failures:
        print(f"error: {len(failures)} SABIO-RK acquisition(s) failed.", file=sys.stderr)
        return 1
    print(f"Saved SABIO-RK raw results for {len(completed)} mapped serum proteases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
