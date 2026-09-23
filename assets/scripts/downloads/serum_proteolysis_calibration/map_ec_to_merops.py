#!/usr/bin/env python3
"""Map peptidase EC numbers to MEROPS codes, for every EC present in BRENDA.

The serum manifest maps 40 EC numbers, which is the panel the calibration was
scoped to. BRENDA holds kinetics for 542 peptidase EC numbers, and a test of the
MEROPS-to-kinetics hypothesis is not restricted to serum. This script builds the
wider mapping so that the extraction can cover every peptidase for which both a
MEROPS matrix and a measurement exist.

Two mapping paths are used and are recorded separately:

``uniprot_human``
    The EC number is queried against reviewed human UniProt entries and the
    MEROPS cross-reference is read from the record. Automatic and reproducible.

``curated``
    A hand-supplied table, for proteases that are not human and therefore have no
    reviewed human record. Each row must carry its own source. Nothing is guessed
    from name similarity.

    This path is **disabled unless ``--curated`` names a file**. The calibration
    covers human enzymes, so bacterial assignments must be requested explicitly;
    they exist for the control-panel work in ``0_08``.

An EC number that neither path resolves is written to the report and excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

from _common import repository_root, write_json


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
PEPTIDASE_EC_PREFIX = "3.4."
REQUEST_PAUSE_SECONDS = 0.34
DEFAULT_TIMEOUT_SECONDS = 60


def peptidase_ec_numbers(archive_path: Path, min_human_entries: int) -> dict[str, dict[str, object]]:
    """List the peptidase EC numbers BRENDA holds human kinetics for.

    :param archive_path: Licensed BRENDA JSON archive.
    :param min_human_entries: Minimum number of human kinetic entries an EC must
        carry to be worth mapping.
    :return: Mapping from EC number to its recommended name and entry counts.
    """

    import ijson

    with tarfile.open(archive_path, "r:*") as archive:
        member = next(item for item in archive.getmembers() if item.name.lower().endswith(".json"))
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("BRENDA JSON member cannot be opened.")

        selected: dict[str, dict[str, object]] = {}
        for ec_number, record in ijson.kvitems(handle, "data"):
            if not ec_number.startswith(PEPTIDASE_EC_PREFIX) or not isinstance(record, dict):
                continue

            ## Count only entries attributable to a human protein record.
            proteins = record.get("protein") or {}
            human_ids = {
                str(key)
                for key, value in proteins.items()
                if isinstance(value, dict) and str(value.get("organism", "")).strip() == "Homo sapiens"
            }
            counts = {}
            for field in ("turnover_number", "km_value", "kcat_km_value"):
                entries = record.get(field) or []
                counts[field] = sum(
                    1
                    for entry in entries
                    if isinstance(entry, dict)
                    and human_ids.intersection(str(value) for value in (entry.get("proteins") or []))
                )
            if sum(counts.values()) < min_human_entries:
                continue
            selected[ec_number] = {
                "recommended_name": str(record.get("recommended_name", "")),
                **{f"n_human_{field}": value for field, value in counts.items()},
            }
    return selected


def query_uniprot_merops(ec_number: str, timeout_seconds: int) -> list[dict[str, str]]:
    """Return the MEROPS cross-references of reviewed human entries for one EC.

    :param ec_number: Full EC number.
    :param timeout_seconds: Socket timeout.
    :return: One record per matching UniProt entry.
    :raises OSError: If the request fails.
    """

    url = f"{UNIPROT_SEARCH_URL}?" + urllib.parse.urlencode(
        {
            "query": f"(ec:{ec_number}) AND (organism_id:9606) AND (reviewed:true)",
            "fields": "accession,protein_name,gene_primary,ec,xref_merops",
            "format": "json",
            "size": "25",
        }
    )
    request = urllib.request.Request(url, headers={"User-Agent": "pep-compass-serum-acquisition/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.load(response)

    records = []
    for entry in payload.get("results", []):
        merops_codes = [
            reference["id"]
            for reference in entry.get("uniProtKBCrossReferences", [])
            if reference.get("database") == "MEROPS"
        ]
        for code in merops_codes:
            genes = entry.get("genes") or []
            gene_symbol = ""
            if genes:
                gene_symbol = genes[0].get("geneName", {}).get("value", "")
            records.append(
                {
                    "uniprot_accession": entry["primaryAccession"],
                    "merops_code": code,
                    "gene_name": gene_symbol,
                    "protein_name": (
                        entry.get("proteinDescription", {})
                        .get("recommendedName", {})
                        .get("fullName", {})
                        .get("value", "")
                    ),
                }
            )
    return records


def read_curated(path: Path | None) -> dict[str, list[dict[str, str]]]:
    """Read the hand-supplied EC to MEROPS assignments.

    :param path: TSV with ``ec_number``, ``merops_code`` and ``source`` columns.
    :return: Mapping from EC number to its curated records.
    :raises ValueError: If a row lacks a source.
    """

    if path is None or not path.is_file():
        return {}
    curated: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not row.get("source"):
                raise ValueError(f"curated row for {row.get('ec_number')} has no source")
            curated.setdefault(row["ec_number"].strip(), []).append(
                {
                    "merops_code": row["merops_code"].strip(),
                    "protein_name": row.get("protein_name", "").strip(),
                    "source": row["source"].strip(),
                }
            )
    return curated


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured parser.
    """

    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=root / "data" / "serum_proteolysis_calibration" / "raw" / "brenda" / "brenda_2026.1_json.tar.gz",
        help="Licensed BRENDA JSON archive.",
    )
    parser.add_argument(
        "--curated",
        type=Path,
        default=None,
        help=(
            "Hand-supplied assignments for proteases without a reviewed human record, "
            "typically bacterial. Off by default: the calibration is restricted to human "
            "enzymes, and these entries exist for the experimental control panel only."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "serum_proteolysis_calibration" / "manifests" / "peptidase_ec_to_merops.tsv",
        help="Destination mapping table.",
    )
    parser.add_argument(
        "--min-human-entries",
        type=int,
        default=1,
        help="Minimum human kinetic entries for an EC number to be mapped.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Build the peptidase EC to MEROPS mapping.

    :param arguments: Command-line arguments excluding program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)
    if not args.archive.is_file():
        print(f"error: BRENDA archive is missing: {args.archive}", file=sys.stderr)
        return 1

    # Candidate EC numbers
    print("Scanning the BRENDA archive for peptidase EC numbers with human kinetics ...")
    candidates = peptidase_ec_numbers(args.archive, args.min_human_entries)
    print(f"peptidase EC numbers to map: {len(candidates)}")
    curated = read_curated(args.curated)
    print(f"curated assignments supplied: {sum(len(values) for values in curated.values())}")

    # Mapping
    rows: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    for index, (ec_number, metadata) in enumerate(sorted(candidates.items()), start=1):
        records: list[dict[str, str]] = []
        source = ""
        try:
            records = query_uniprot_merops(ec_number, args.timeout_seconds)
            source = "uniprot_human"
        except Exception as error:  # noqa: BLE001 - one failed EC must not abort the run
            print(f"  {ec_number}: UniProt query failed ({error})", file=sys.stderr)
        time.sleep(REQUEST_PAUSE_SECONDS)

        if not records and ec_number in curated:
            records = curated[ec_number]
            source = "curated"

        if not records:
            unresolved.append({"ec_number": ec_number, **metadata})
            continue

        for record in records:
            rows.append(
                {
                    "ec_number": ec_number,
                    "merops_code": record["merops_code"],
                    "uniprot_accession": record.get("uniprot_accession", ""),
                    "gene_name": record.get("gene_name", ""),
                    "protein_name": record.get("protein_name", ""),
                    "mapping_source": record.get("source", source),
                    "brenda_recommended_name": metadata["recommended_name"],
                    "n_human_turnover_number": metadata["n_human_turnover_number"],
                    "n_human_km_value": metadata["n_human_km_value"],
                    "n_human_kcat_km_value": metadata["n_human_kcat_km_value"],
                }
            )
        if index % 25 == 0:
            print(f"  mapped {index}/{len(candidates)} EC numbers")

    # Outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ec_number", "merops_code", "uniprot_accession", "gene_name", "protein_name", "mapping_source",
        "brenda_recommended_name", "n_human_turnover_number", "n_human_km_value",
        "n_human_kcat_km_value",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    report_path = args.output.with_name("peptidase_ec_unresolved.json")
    write_json(
        report_path,
        {
            "n_candidate_ec": len(candidates),
            "n_mapped_ec": len({row["ec_number"] for row in rows}),
            "n_unresolved_ec": len(unresolved),
            "unresolved": unresolved,
        },
    )

    print(f"\nmapped EC numbers: {len({row['ec_number'] for row in rows})} / {len(candidates)}")
    print(f"MEROPS codes reached: {len({row['merops_code'] for row in rows})}")
    print(f"unresolved EC numbers: {len(unresolved)} -> {report_path}")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
