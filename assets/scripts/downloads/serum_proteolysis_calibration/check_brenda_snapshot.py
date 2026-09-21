#!/usr/bin/env python3
"""Validate the manually licensed BRENDA JSON snapshot and write its receipt."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

from _common import acquisition_root, raw_file_record, write_receipt


EXPECTED_RELEASE = "2026.1"


def read_header(archive_path: Path) -> dict[str, Any]:
    """Read the top-level JSON object from a one-member BRENDA archive.

    :param archive_path: Licensed BRENDA JSON tar archive.
    :return: Parsed top-level BRENDA object.
    :raises ValueError: If the archive does not contain one JSON file.
    """

    with tarfile.open(archive_path, "r:*") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.lower().endswith(".json")]
        if len(members) != 1:
            raise ValueError("BRENDA archive must contain exactly one JSON member.")
        handle = archive.extractfile(members[0])
        if handle is None:
            raise ValueError("BRENDA JSON member cannot be opened.")
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("BRENDA JSON top-level value is not an object.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=acquisition_root() / "raw" / "brenda" / "brenda_2026.1_json.tar.gz",
        help="Installed BRENDA JSON archive.",
    )
    parser.add_argument("--release", default=EXPECTED_RELEASE, help="Required BRENDA release.")
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Validate the local BRENDA snapshot without extracting or modifying it.

    :param arguments: Command-line arguments excluding program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)
    if not args.archive.is_file() or not tarfile.is_tarfile(args.archive):
        print(f"error: BRENDA archive is missing or invalid: {args.archive}", file=sys.stderr)
        return 1
    try:
        header = read_header(args.archive)
    except (OSError, ValueError, tarfile.TarError, json.JSONDecodeError) as error:
        print(f"error: cannot validate BRENDA archive: {error}", file=sys.stderr)
        return 1

    if header.get("release") != args.release:
        print(
            f"error: BRENDA release is {header.get('release')!r}, expected {args.release!r}.",
            file=sys.stderr,
        )
        return 1
    if not isinstance(header.get("version"), str) or not isinstance(header.get("data"), dict):
        print("error: BRENDA JSON does not have the required release/version/data top-level structure.", file=sys.stderr)
        return 1

    write_receipt(
        args.archive.parent / "download_manifest.json",
        "BRENDA",
        args.release,
        "manual licensed snapshot",
        [raw_file_record(args.archive, "manual licensed snapshot", "validated")],
    )
    print(f"Validated BRENDA {args.release} JSON snapshot: {args.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
