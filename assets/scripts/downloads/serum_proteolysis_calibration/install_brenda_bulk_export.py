#!/usr/bin/env python3
"""Install a licensed, manually downloaded BRENDA bulk archive with provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path


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


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured argument parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="BRENDA .tar.gz archive obtained after accepting the official licence.")
    parser.add_argument(
        "--format",
        required=True,
        choices=("json", "text"),
        help="Format selected on the BRENDA bulk-download page.",
    )
    parser.add_argument(
        "--release",
        required=True,
        help="BRENDA release shown on the bulk-download page, for example 2026.1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root() / "data" / "serum_proteolysis_calibration" / "raw" / "brenda",
        help="Directory for the original archive and provenance record.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing archive with the same release and format.")
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Install one BRENDA bulk archive without extracting or parsing it.

    :param arguments: Command-line arguments excluding the program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)
    if not args.archive.is_file():
        print(f"error: archive does not exist or is not a file: {args.archive}", file=sys.stderr)
        return 2
    if not tarfile.is_tarfile(args.archive):
        print("error: archive is not a readable tar archive.", file=sys.stderr)
        return 2

    destination = args.output_dir / f"brenda_{args.release}_{args.format}.tar.gz"
    provenance_path = destination.with_suffix(destination.suffix + ".provenance.json")
    if destination.exists() and not args.force:
        print(f"BRENDA archive already present: {destination}")
        print("Use --force to replace it.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".part")
    try:
        shutil.copyfile(args.archive, temporary_path)
        temporary_path.replace(destination)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        print(f"error: could not install BRENDA archive: {error}", file=sys.stderr)
        return 1

    provenance_path.write_text(
        json.dumps(
            {
                "source_database": "BRENDA",
                "release": args.release,
                "format": args.format,
                "retrieved_utc": datetime.now(UTC).isoformat(),
                "source_archive_name": args.archive.name,
                "local_path": str(destination),
                "sha256": sha256sum(destination),
                "license_acceptance": "Accepted manually on the official BRENDA bulk-download page.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Installed BRENDA bulk archive: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
