#!/usr/bin/env python3
"""Build the fixed human-serum protease query manifest from MEROPS metadata."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from _common import acquisition_root, repository_root, utc_now, write_json


MANIFEST_COLUMNS = (
    "merops_code",
    "merops_name",
    "merops_family",
    "n_merops_cleavages",
    "serum_category",
    "include_in_hydrolysis_model",
    "uniprot_accession",
    "gene_name",
    "ensembl_gene_id",
    "ec_number",
    "mapping_status",
)
NON_HYDROLASE_CODE = "C111.001"


def build_rows(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Convert the local MEROPS JSON payload to query-manifest rows.

    :param payload: Parsed ``codes.json`` payload.
    :return: Rows in MEROPS matrix order.
    :raises ValueError: If required MEROPS entry fields are absent.
    """

    peptidases = payload.get("peptidases")
    if not isinstance(peptidases, list) or not peptidases:
        raise ValueError("MEROPS payload has no non-empty 'peptidases' list.")

    rows: list[dict[str, str]] = []
    for entry in sorted(peptidases, key=lambda value: value.get("row", -1)):
        if not isinstance(entry, dict):
            raise ValueError("MEROPS peptidase entry is not an object.")
        code = entry.get("code")
        name = entry.get("name")
        if not isinstance(code, str) or not code or not isinstance(name, str) or not name:
            raise ValueError("Each MEROPS peptidase requires a non-empty code and name.")
        rows.append(
            {
                "merops_code": code,
                "merops_name": name,
                "merops_family": code.partition(".")[0],
                "n_merops_cleavages": str(entry.get("n_cleavages", "")),
                "serum_category": str(entry.get("serum_category", "")),
                "include_in_hydrolysis_model": "false" if code == NON_HYDROLASE_CODE else "true",
                "uniprot_accession": "",
                "gene_name": "",
                "ensembl_gene_id": "",
                "ec_number": "",
                "mapping_status": "unmapped",
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write fixed-order serum protease rows to TSV.

    :param path: TSV destination.
    :param rows: Manifest rows.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merops-codes",
        type=Path,
        default=repository_root() / "data" / "merops" / "35_human_serum" / "codes.json",
        help="Authoritative local MEROPS serum-panel metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=acquisition_root() / "manifests" / "serum_proteases.tsv",
        help="Output query-manifest TSV.",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Build the serum protease query manifest.

    :param arguments: Command-line arguments excluding program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)
    try:
        payload = json.loads(args.merops_codes.read_text(encoding="utf-8"))
        rows = build_rows(payload)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: could not build the MEROPS manifest: {error}", file=sys.stderr)
        return 1

    write_tsv(args.output, rows)
    write_json(
        args.output.with_suffix(".build.json"),
        {
            "source": "MEROPS local dataset 35",
            "source_path": str(args.merops_codes),
            "created_at": utc_now(),
            "n_entries": len(rows),
            "n_included": sum(row["include_in_hydrolysis_model"] == "true" for row in rows),
            "excluded_merops_codes": [NON_HYDROLASE_CODE],
        },
    )
    print(f"Wrote {len(rows)} MEROPS rows ({len(rows) - 1} included) to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
