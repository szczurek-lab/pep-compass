#!/usr/bin/env python3
"""Retrieve only Human serum and Human plasma records from PEPlife2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from _common import DEFAULT_TIMEOUT_SECONDS, acquisition_root, merge_receipt, raw_file_record, write_bytes


PEPLIFE2_API_URL = "https://webs.iiitd.edu.in/raghava/peplife2/api/api.php"
MEDIA = {
    "human_serum": "Human serum",
    "human_plasma": "Human plasma",
}


def get_json(url: str, timeout_seconds: int) -> tuple[dict[str, Any], bytes]:
    """Retrieve one PEPlife2 JSON response.

    :param url: Fully qualified PEPlife2 request URL.
    :param timeout_seconds: Network timeout in seconds.
    :return: Decoded response object and its unmodified source bytes.
    :raises ValueError: If the API returns an invalid or not-found payload.
    """

    request = Request(url, headers={"Accept": "application/json", "User-Agent": "pep-compass-serum-acquisition/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        raw_payload = response.read()
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise ValueError("PEPlife2 did not return a JSON object.")
    if str(payload.get("status", "")) == "404":
        raise ValueError("PEPlife2 reported status=404 for the requested medium.")
    return payload, raw_payload


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("medium", choices=sorted(MEDIA), help="PEPlife2 medium to retrieve.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=acquisition_root() / "raw" / "peplife2",
        help="Directory for unmodified PEPlife2 JSON responses.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing response.")
    parser.add_argument("--base-url", default=PEPLIFE2_API_URL, help=argparse.SUPPRESS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help=argparse.SUPPRESS)
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Retrieve one restricted PEPlife2 medium.

    :param arguments: Command-line arguments excluding program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)
    medium = MEDIA[args.medium]
    url = f"{args.base_url}?{urlencode({'dataType': 'organism', 'dataValue': medium})}"
    destination = args.output_dir / f"{args.medium}.json"
    try:
        if destination.exists() and not args.force:
            status = "retained"
        else:
            _, raw_payload = get_json(url, args.timeout_seconds)
            write_bytes(destination, raw_payload)
            status = "downloaded"
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: PEPlife2 {args.medium} request failed: {error}", file=sys.stderr)
        return 1

    merge_receipt(
        args.output_dir / "download_manifest.json",
        "PEPlife2",
        args.medium,
        url,
        [raw_file_record(destination, url, status)],
    )
    print(f"{status.capitalize()} PEPlife2 response: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
