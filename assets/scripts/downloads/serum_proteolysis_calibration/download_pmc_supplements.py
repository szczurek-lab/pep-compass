#!/usr/bin/env python3
"""Download whitelisted numerical supplements from one PMC AWS Open Data article."""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from _common import DEFAULT_TIMEOUT_SECONDS, acquisition_root, download_url, raw_file_record, write_json, write_receipt


PMC_BUCKET_URL = "https://pmc-oa-opendata.s3.amazonaws.com"
ALLOWED_SUFFIXES = (".xlsx", ".xls", ".csv", ".tsv", ".txt", ".zip", ".gz", ".fasta", ".fa")
DATASET_PROFILES = {
    "mmp_pics": {
        "pmcid": "PMC4777984",
        "destination": ("mmp_pics", "PMC4777984"),
        "patterns": (r"(?i).*\.(?:xlsx?|zip)$",),
    },
    "dpp4_qpisa": {
        "pmcid": "PMC11612144",
        "destination": ("dpp4_qpisa", "PMC11612144"),
        "patterns": (r"(?i).*\.(?:xlsx?|csv|tsv|zip)$",),
    },
    "qmsp_ms": {
        "pmcid": "PMC6495252",
        "destination": ("qmsp_ms", "PMC6495252"),
        "patterns": (r"(?i).*\.(?:xlsx?|csv|tsv|txt|zip|fasta|fa)$",),
    },
    "adamts13": {
        "pmcid": "PMC4522773",
        "destination": ("adamts13",),
        "patterns": (r"(?i).*pnas\.1511328112\.sd01\.xlsx$",),
    },
}


def get_xml(url: str, timeout_seconds: int) -> element_tree.Element:
    """Retrieve and parse one unsigned S3 XML listing.

    :param url: S3 listing URL.
    :param timeout_seconds: Network timeout in seconds.
    :return: XML root element.
    :raises ValueError: If the response is not parseable XML.
    """

    request = Request(url, headers={"Accept": "application/xml", "User-Agent": "pep-compass-serum-acquisition/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
    try:
        return element_tree.fromstring(payload)
    except element_tree.ParseError as error:
        raise ValueError("PMC Open Data listing did not return XML.") from error


def find_texts(root: element_tree.Element, element_name: str) -> list[str]:
    """Extract non-empty values from a namespace-agnostic XML element name.

    :param root: XML root.
    :param element_name: Local XML element name.
    :return: Extracted text values.
    """

    return [
        element.text
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == element_name and isinstance(element.text, str) and element.text
    ]


def list_article_prefixes(bucket_url: str, pmcid: str, timeout_seconds: int) -> tuple[str, list[str]]:
    """Resolve a PMC article-version prefix without listing the full bucket.

    :param bucket_url: Public PMC S3 bucket URL.
    :param pmcid: Article PMCID.
    :param timeout_seconds: Network timeout in seconds.
    :return: Listing URL and matching article-version prefixes.
    """

    url = f"{bucket_url}?{urlencode({'list-type': '2', 'prefix': f'{pmcid}.', 'delimiter': '/'})}"
    root = get_xml(url, timeout_seconds)
    root_prefix = f"{pmcid}."
    prefixes = [prefix for prefix in find_texts(root, "Prefix") if prefix != root_prefix and prefix.startswith(root_prefix)]
    return url, sorted(set(prefixes))


def list_objects(bucket_url: str, prefix: str, timeout_seconds: int) -> tuple[str, list[str]]:
    """List one resolved article prefix through the unsigned S3 API.

    :param bucket_url: Public PMC S3 bucket URL.
    :param prefix: Resolved article-version prefix.
    :param timeout_seconds: Network timeout in seconds.
    :return: Listing URL and object keys.
    """

    url = f"{bucket_url}?{urlencode({'list-type': '2', 'prefix': prefix})}"
    root = get_xml(url, timeout_seconds)
    return url, find_texts(root, "Key")


def select_objects(keys: list[str], patterns: tuple[str, ...]) -> list[str]:
    """Select numerical supplements from a resolved PMC object list.

    :param keys: S3 object keys under one article prefix.
    :param patterns: Filename regular-expression allowlist.
    :return: Selected object keys.
    """

    compiled = [re.compile(pattern) for pattern in patterns]
    selected: list[str] = []
    for key in keys:
        name = Path(key).name
        if not name.lower().endswith(ALLOWED_SUFFIXES):
            continue
        if any(pattern.search(name) for pattern in compiled):
            selected.append(key)
    return sorted(set(selected))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASET_PROFILES), help="Whitelisted PMC supplement profile.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=acquisition_root() / "raw",
        help="Root directory for source artifacts.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing source files.")
    parser.add_argument("--bucket-url", default=PMC_BUCKET_URL, help=argparse.SUPPRESS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help=argparse.SUPPRESS)
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Download selected numerical supplements for one source profile.

    :param arguments: Command-line arguments excluding program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)
    profile = DATASET_PROFILES[args.dataset]
    try:
        prefix_listing_url, prefixes = list_article_prefixes(args.bucket_url, profile["pmcid"], args.timeout_seconds)
        if len(prefixes) != 1:
            raise ValueError(f"Expected one PMC article-version prefix for {profile['pmcid']}, found {prefixes!r}.")
        object_listing_url, keys = list_objects(args.bucket_url, prefixes[0], args.timeout_seconds)
        selected = select_objects(keys, profile["patterns"])
        if not selected:
            raise ValueError(f"No allowed numerical supplement matched profile {args.dataset!r}.")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        print(f"error: could not inspect PMC supplements for {args.dataset}: {error}", file=sys.stderr)
        return 1

    destination_dir = args.raw_root.joinpath(*profile["destination"])
    files: list[dict[str, object]] = []
    try:
        for key in selected:
            filename = Path(key).name
            destination = destination_dir / filename
            source_url = f"{args.bucket_url.rstrip('/')}/{key}"
            if destination.exists() and not args.force:
                status = "retained"
            else:
                print(f"Downloading {filename}")
                download_url(source_url, destination, args.timeout_seconds)
                status = "downloaded"
            files.append(raw_file_record(destination, source_url, status))
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        print(f"error: could not download PMC supplement: {error}", file=sys.stderr)
        return 1

    write_json(
        destination_dir / "selection.json",
        {
            "dataset": args.dataset,
            "pmcid": profile["pmcid"],
            "article_prefix_listing_url": prefix_listing_url,
            "object_listing_url": object_listing_url,
            "article_prefix": prefixes[0],
            "selected_object_keys": selected,
        },
    )
    files.append(raw_file_record(destination_dir / "selection.json", object_listing_url, "generated_provenance"))
    write_receipt(
        destination_dir / "download_manifest.json",
        f"PMC Open Data: {args.dataset}",
        profile["pmcid"],
        object_listing_url,
        files,
    )
    print(f"Saved {len(selected)} PMC supplement file(s) under: {destination_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
