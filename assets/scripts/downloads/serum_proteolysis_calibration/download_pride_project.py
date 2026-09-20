#!/usr/bin/env python3
"""Download selected processed files and metadata from a PRIDE Archive project."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from _common import acquisition_root, raw_file_record, write_bytes, write_receipt


DEFAULT_PRIDE_API_BASE_URL = "https://www.ebi.ac.uk/pride/ws/archive/v2"
DEFAULT_TIMEOUT_SECONDS = 60
RAW_MS_SUFFIXES = (".raw", ".mzml", ".mzxml", ".wiff")
PROFILES = {
    "mmp_pics": (r"(?i)\.tandem\.pep\.xml$",),
    "dpp4_qpisa": (
        r"(?i)\.rds$",
        r"(?i).*ExpDesign.*\.txt$",
        r"(?i).*PeptideGroups.*\.txt$",
    ),
}


def repository_root() -> Path:
    """Return the repository root inferred from this script location.

    :return: Absolute repository root.
    """

    return Path(__file__).resolve().parents[4]


def sha256sum(path: Path) -> str:
    """Return the SHA-256 checksum of ``path``.

    :param path: File to hash.
    :return: Lowercase hexadecimal SHA-256 digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(url: str, timeout_seconds: int) -> tuple[list[dict[str, Any]], bytes]:
    """Retrieve a JSON array from a PRIDE Archive endpoint.

    :param url: Fully qualified endpoint URL.
    :param timeout_seconds: Network timeout in seconds.
    :return: Decoded PRIDE file records and unmodified response bytes.
    :raises ValueError: If the response is not a JSON array.
    """

    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout_seconds) as response:
        raw_payload = response.read()
    payload = json.loads(raw_payload)

    if not isinstance(payload, list):
        raise ValueError("PRIDE files endpoint did not return a JSON array.")
    return payload, raw_payload


def select_download_url(record: dict[str, Any]) -> str:
    """Select an HTTPS public location from one PRIDE file record.

    PRIDE commonly exposes FTP and Aspera locations. The public FTP location
    has an equivalent HTTPS URL, which avoids requiring an FTP client.

    :param record: PRIDE API file record.
    :return: Public HTTP(S) URL.
    :raises ValueError: If no HTTP(S) or FTP public location is present.
    """

    locations = record.get("publicFileLocations", [])
    if not isinstance(locations, list):
        raise ValueError("PRIDE file record has an invalid publicFileLocations field.")

    for location in locations:
        value = location.get("value") if isinstance(location, dict) else None
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            return value

    for location in locations:
        value = location.get("value") if isinstance(location, dict) else None
        if isinstance(value, str) and value.startswith("ftp://"):
            return "https://" + value.removeprefix("ftp://")

    file_name = record.get("fileName", "<unknown>")
    raise ValueError(f"No public HTTP(S) or FTP location for PRIDE file {file_name!r}.")


def safe_file_name(record: dict[str, Any]) -> str:
    """Validate and return the basename declared by a PRIDE file record.

    :param record: PRIDE API file record.
    :return: Safe local filename.
    :raises ValueError: If the name is absent or contains a path component.
    """

    file_name = record.get("fileName")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("PRIDE file record has no fileName.")
    if Path(file_name).name != file_name:
        raise ValueError(f"Refusing a PRIDE filename with a path component: {file_name!r}.")
    return file_name


def download_file(url: str, destination: Path, timeout_seconds: int) -> str:
    """Download one file atomically and return its SHA-256 checksum.

    :param url: Public source URL.
    :param destination: Final local path.
    :param timeout_seconds: Network timeout in seconds.
    :return: Lowercase hexadecimal SHA-256 digest of the downloaded file.
    """

    temporary_path = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    request = Request(url, headers={"User-Agent": "pep-compass-data-download/1.0"})

    try:
        with urlopen(request, timeout=timeout_seconds) as response, temporary_path.open("wb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
                digest.update(chunk)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured argument parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accession", help="Public PRIDE project accession, for example PXD042089.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=acquisition_root() / "raw" / "pride",
        help="Directory containing one subdirectory per PRIDE accession.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="REGEX",
        help="Regular expression matched against file names. Repeat to select a union of patterns.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        help="Dataset-specific processed-file allowlist. Cannot be combined with --include.",
    )
    parser.add_argument(
        "--allow-raw-ms",
        action="store_true",
        help="Permit an explicit --include pattern to select RAW/MS files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing local files instead of retaining them.",
    )
    parser.add_argument("--base-url", default=DEFAULT_PRIDE_API_BASE_URL, help=argparse.SUPPRESS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help=argparse.SUPPRESS)
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Run the PRIDE downloader.

    :param arguments: Command-line arguments excluding the program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)
    accession = args.accession.upper()
    if not re.fullmatch(r"PXD\d+", accession):
        print("error: accession must have the form PXD followed by digits.", file=sys.stderr)
        return 2
    if args.profile and args.include:
        print("error: --profile and --include cannot be used together.", file=sys.stderr)
        return 2

    try:
        configured_patterns = PROFILES[args.profile] if args.profile else args.include
        patterns = [re.compile(pattern) for pattern in configured_patterns]
    except re.error as error:
        print(f"error: invalid --include regular expression: {error}", file=sys.stderr)
        return 2

    api_url = f"{args.base_url.rstrip('/')}/projects/{accession}/files/all"
    try:
        records, raw_manifest = read_json(api_url, args.timeout_seconds)
    except (HTTPError, URLError, TimeoutError, ValueError) as error:
        print(f"error: could not retrieve PRIDE metadata from {api_url}: {error}", file=sys.stderr)
        return 1

    if not records:
        print(
            f"error: PRIDE returned no files for {accession}. This accession may be hosted in another "
            "ProteomeXchange repository; check the source accession before retrying.",
            file=sys.stderr,
        )
        return 1

    project_dir = args.output_dir / accession
    files_dir = project_dir / "files"
    project_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = project_dir / "manifest.json"
    write_bytes(manifest_path, raw_manifest)
    receipt_files = [raw_file_record(manifest_path, api_url, "downloaded")]

    if not patterns:
        print(f"Saved PRIDE manifest for {accession}: {manifest_path}")
        print("No project files were downloaded. Use --profile or --include REGEX explicitly.")
        write_receipt(
            project_dir / "download_manifest.json",
            "PRIDE Archive",
            accession,
            api_url,
            receipt_files,
        )
        return 0

    selected_records = [
        record
        for record in records
        if any(pattern.search(safe_file_name(record)) for pattern in patterns)
    ]
    raw_records = [record for record in selected_records if safe_file_name(record).lower().endswith(RAW_MS_SUFFIXES)]
    if raw_records and not args.allow_raw_ms:
        raw_names = ", ".join(safe_file_name(record) for record in raw_records[:3])
        print(
            f"error: selected raw mass-spectrometry files ({raw_names}). "
            "Use --allow-raw-ms only for a justified reprocessing workflow.",
            file=sys.stderr,
        )
        return 2
    if not selected_records:
        print("error: no PRIDE files matched the requested selection.", file=sys.stderr)
        return 1

    files_dir.mkdir(exist_ok=True)
    for record in selected_records:
        file_name = safe_file_name(record)
        destination = files_dir / file_name
        source_url = select_download_url(record)

        if destination.exists() and not args.force:
            checksum = sha256sum(destination)
            status = "retained"
        else:
            print(f"Downloading {file_name}")
            try:
                checksum = download_file(source_url, destination, args.timeout_seconds)
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                print(f"error: could not download {file_name}: {error}", file=sys.stderr)
                return 1
            status = "downloaded"

        receipt_files.append(raw_file_record(destination, source_url, status))

    write_receipt(
        project_dir / "download_manifest.json",
        "PRIDE Archive",
        accession,
        api_url,
        receipt_files,
    )
    print(f"Saved {len(selected_records)} PRIDE file(s) under: {files_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
