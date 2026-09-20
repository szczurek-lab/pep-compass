#!/usr/bin/env python3
"""Download only the fixed DRAMP and Human Protein Atlas source files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

from _common import DEFAULT_TIMEOUT_SECONDS, acquisition_root, download_url, merge_receipt, raw_file_record


SOURCES = {
    "dramp_stability": {
        "url": "https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/stability_amps.xlsx",
        "destination": ("dramp", "stability_amps.xlsx"),
        "version": "DRAMP stability_amps.xlsx",
    },
    "hpa_blood_ms": {
        "url": "https://www.proteinatlas.org/download/tsv/blood_ms_concentration.tsv.zip",
        "destination": ("hpa", "blood_ms_concentration.tsv.zip"),
        "version": "Human Protein Atlas blood MS concentration",
    },
    "hpa_blood_immunoassay": {
        "url": "https://www.proteinatlas.org/download/tsv/blood_immunoassay_concentration.tsv.zip",
        "destination": ("hpa", "blood_concentration_immunoassay.tsv.zip"),
        "version": "Human Protein Atlas blood immunoassay concentration",
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=sorted(SOURCES), help="Fixed source to retrieve.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=acquisition_root() / "raw",
        help="Root directory for source artifacts.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing local source artifact.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help=argparse.SUPPRESS)
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Download one fixed, whitelisted source artifact.

    :param arguments: Command-line arguments excluding program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)
    source = SOURCES[args.source]
    destination = args.raw_root.joinpath(*source["destination"])
    receipt_path = destination.parent / "download_manifest.json"

    try:
        if destination.exists() and not args.force:
            status = "retained"
        else:
            print(f"Downloading {args.source}: {source['url']}")
            download_url(source["url"], destination, args.timeout_seconds)
            status = "downloaded"
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        print(f"error: could not download {args.source}: {error}", file=sys.stderr)
        return 1

    receipt_source = "Human Protein Atlas" if args.source.startswith("hpa_") else "DRAMP"
    merge_receipt(
        receipt_path,
        receipt_source,
        source["version"],
        source["url"],
        [raw_file_record(destination, source["url"], status)],
    )
    print(f"{status.capitalize()} fixed source: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
