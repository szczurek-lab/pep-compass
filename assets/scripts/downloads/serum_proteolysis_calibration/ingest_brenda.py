#!/usr/bin/env python3
"""Extract kinetic parameters for the serum protease panel from the BRENDA snapshot.

The licensed BRENDA release is distributed as a single ~677 MB JSON document
inside a one-member tar archive. This script streams that document, keeps only
the EC numbers listed in the serum protease manifest, and writes one compact
table of raw kinetic entries plus a per-EC coverage summary.

Values and units are preserved exactly as BRENDA reports them; unit conversion
and substrate resolution belong to the later normalization stage, not here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from _common import acquisition_root, raw_file_record, write_json, write_receipt


# BRENDA reports each parameter family in a fixed unit, documented per field.
# REMARK: These strings are provenance, not a conversion instruction. Normalization
# to SI (M, 1/s, 1/(M*s)) happens downstream, where measurement context is known.
PARAMETER_UNITS: dict[str, str] = {
    "turnover_number": "1/s",
    "km_value": "mM",
    "kcat_km_value": "1/(mM*s)",
}

TARGET_ORGANISM = "Homo sapiens"

# "2.6 {L-2-NaI-7-amido-4-carbamoylmethylcoumarin}" -> value, substrate.
VALUE_PATTERN = re.compile(r"^\s*(?P<value>[^{}]*?)\s*(?:\{(?P<substrate>.*)\})?\s*$", re.DOTALL)
NUMBER_PATTERN = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$")
RANGE_PATTERN = re.compile(
    r"^(?P<low>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*-\s*(?P<high>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$"
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read the serum protease manifest.

    :param path: Manifest TSV produced by ``build_serum_manifest.py``.
    :return: Manifest rows as dictionaries.
    :raises ValueError: If the manifest lacks the required columns.
    """

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"merops_code", "ec_number", "include_in_hydrolysis_model", "uniprot_accession"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError(f"Manifest {path} does not contain the required columns {sorted(required)}.")
    return rows


def read_ec_mapping(path: Path) -> list[dict[str, str]]:
    """Read the wider peptidase EC to MEROPS mapping as manifest-shaped rows.

    The serum manifest covers the 40 EC numbers of the serum panel. This mapping,
    produced by ``map_ec_to_merops.py``, covers every peptidase EC number BRENDA
    holds human kinetics for, and is normalized here into the same row shape so
    that one extraction path serves both scopes.

    :param path: Mapping TSV produced by ``map_ec_to_merops.py``.
    :return: Rows carrying ``merops_code``, ``ec_number``, ``uniprot_accession``,
        ``gene_name`` and ``include_in_hydrolysis_model``.
    :raises ValueError: If the mapping lacks the required columns.
    """

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"ec_number", "merops_code"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError(f"Mapping {path} does not contain the required columns {sorted(required)}.")
    return [
        {
            "merops_code": row["merops_code"],
            "ec_number": row["ec_number"],
            "uniprot_accession": row.get("uniprot_accession", ""),
            "gene_name": row.get("protein_name", ""),
            "include_in_hydrolysis_model": "true",
            "mapping_source": row.get("mapping_source", ""),
        }
        for row in rows
    ]


def build_ec_index(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group included manifest proteases by EC number.

    One EC number can carry several MEROPS codes (isoenzymes sharing an activity),
    so the index maps each EC to every protease that requested it.

    :param rows: Manifest rows.
    :return: Mapping from EC number to the manifest rows requesting it.
    """

    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("include_in_hydrolysis_model") != "true":
            continue
        ec_number = (row.get("ec_number") or "").strip()
        if ec_number:
            index[ec_number].append(row)
    return dict(index)


def split_value(raw_value: str) -> tuple[str, str | None]:
    """Split a BRENDA parameter string into its value part and substrate part.

    :param raw_value: Raw ``value`` field, e.g. ``"2.6 {L-Ala-4-nitroanilide}"``.
    :return: Tuple of value text and substrate text (``None`` when absent).
    """

    match = VALUE_PATTERN.match(raw_value or "")
    if match is None:
        return (raw_value or "").strip(), None
    substrate = match.group("substrate")
    return match.group("value").strip(), substrate.strip() if substrate is not None else None


def parse_numeric(value_text: str) -> tuple[float | None, float | None, float | None, str]:
    """Interpret the value part of a BRENDA entry.

    BRENDA mixes single numbers, ranges and free text (``additional information``)
    in the same field, so the numeric interpretation is explicit and flagged.

    :param value_text: Value part returned by :func:`split_value`.
    :return: Tuple of point value, range low, range high and value kind.
    """

    text = value_text.strip()
    if NUMBER_PATTERN.match(text):
        return float(text), None, None, "numeric"
    range_match = RANGE_PATTERN.match(text)
    if range_match is not None:
        low = float(range_match.group("low"))
        high = float(range_match.group("high"))
        return None, low, high, "range"
    return None, None, None, "non_numeric"


def iter_ec_records(archive_path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Stream ``(ec_number, record)`` pairs from the BRENDA JSON archive.

    :param archive_path: Licensed BRENDA JSON tar archive.
    :return: Iterator over EC records.
    :raises ValueError: If the archive does not hold exactly one JSON member.
    """

    import ijson

    with tarfile.open(archive_path, "r:*") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.lower().endswith(".json")]
        if len(members) != 1:
            raise ValueError("BRENDA archive must contain exactly one JSON member.")
        handle = archive.extractfile(members[0])
        if handle is None:
            raise ValueError("BRENDA JSON member cannot be opened.")
        yield from ijson.kvitems(handle, "data")


def collect_protein_organisms(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the protein block of one EC record by protein id.

    :param record: BRENDA EC record.
    :return: Mapping from protein id to its metadata.
    """

    protein_block = record.get("protein")
    if not isinstance(protein_block, dict):
        return {}
    return {str(key): value for key, value in protein_block.items() if isinstance(value, dict)}


def collect_reference_pmids(record: dict[str, Any]) -> dict[str, str]:
    """Index the reference block of one EC record by reference id.

    :param record: BRENDA EC record.
    :return: Mapping from reference id to its PMID (empty when not reported).
    """

    reference_block = record.get("reference")
    if not isinstance(reference_block, dict):
        return {}
    pmids: dict[str, str] = {}
    for key, value in reference_block.items():
        if isinstance(value, dict) and value.get("pmid") is not None:
            pmids[str(key)] = str(value["pmid"])
    return pmids


def extract_entries(
    ec_number: str,
    record: dict[str, Any],
    manifest_rows: list[dict[str, str]],
    organism: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract kinetic entries of one EC record for the requested organism.

    :param ec_number: EC number of the record.
    :param record: BRENDA EC record.
    :param manifest_rows: Manifest rows requesting this EC number.
    :param organism: Organism whose protein records are kept.
    :return: Tuple of extracted rows and the per-EC coverage summary.
    """

    ## Resolve the organism-specific protein identifiers of this EC record
    proteins = collect_protein_organisms(record)
    pmids = collect_reference_pmids(record)
    target_protein_ids = {
        protein_id
        for protein_id, protein in proteins.items()
        if str(protein.get("organism", "")).strip() == organism
    }

    merops_codes = ";".join(sorted(row["merops_code"] for row in manifest_rows))
    accessions = ";".join(sorted({row["uniprot_accession"] for row in manifest_rows if row.get("uniprot_accession")}))
    gene_names = ";".join(sorted({row.get("gene_name", "") for row in manifest_rows if row.get("gene_name")}))

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "ec_number": ec_number,
        "merops_codes": merops_codes,
        "n_protein_records": len(proteins),
        "n_target_protein_records": len(target_protein_ids),
    }

    ## Walk the three kinetic parameter families
    for parameter_type, unit in PARAMETER_UNITS.items():
        entries = record.get(parameter_type)
        entries = entries if isinstance(entries, list) else []
        summary[f"n_{parameter_type}_all_organisms"] = len(entries)

        kept = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            ### Keep only entries attributable to the target organism
            entry_protein_ids = [str(value) for value in entry.get("proteins", []) or []]
            if not target_protein_ids.intersection(entry_protein_ids):
                continue

            value_text, substrate = split_value(str(entry.get("value", "")))
            point_value, range_low, range_high, value_kind = parse_numeric(value_text)
            entry_reference_ids = [str(value) for value in entry.get("references", []) or []]

            rows.append(
                {
                    "ec_number": ec_number,
                    "merops_codes": merops_codes,
                    "uniprot_accessions": accessions,
                    "gene_names": gene_names,
                    "organism": organism,
                    "parameter_type": parameter_type,
                    "unit_raw": unit,
                    "value_raw": str(entry.get("value", "")),
                    "value_numeric": point_value,
                    "value_range_low": range_low,
                    "value_range_high": range_high,
                    "value_kind": value_kind,
                    "substrate_raw": substrate,
                    "comment_raw": str(entry.get("comment", "")),
                    "protein_ids": ";".join(entry_protein_ids),
                    "reference_ids": ";".join(entry_reference_ids),
                    "pmids": ";".join(
                        sorted({pmids[reference_id] for reference_id in entry_reference_ids if reference_id in pmids})
                    ),
                }
            )
            kept += 1

        summary[f"n_{parameter_type}_target_organism"] = kept

    return rows, summary


def write_table(rows: list[dict[str, Any]], destination: Path) -> None:
    """Write the extracted entries as a Parquet table.

    :param rows: Extracted kinetic entries.
    :param destination: Destination Parquet file.
    """

    import pandas as pd

    columns = [
        "ec_number",
        "merops_codes",
        "uniprot_accessions",
        "gene_names",
        "organism",
        "parameter_type",
        "unit_raw",
        "value_raw",
        "value_numeric",
        "value_range_low",
        "value_range_high",
        "value_kind",
        "substrate_raw",
        "comment_raw",
        "protein_ids",
        "reference_ids",
        "pmids",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=acquisition_root() / "raw" / "brenda" / "brenda_2026.1_json.tar.gz",
        help="Licensed BRENDA JSON archive validated by check_brenda_snapshot.py.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=acquisition_root() / "manifests" / "serum_proteases.tsv",
        help="Serum protease manifest restricting the extracted EC numbers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=acquisition_root() / "processed" / "kinetics" / "brenda_serum_proteases.parquet",
        help="Destination Parquet table of raw BRENDA kinetic entries.",
    )
    parser.add_argument(
        "--ec-mapping",
        type=Path,
        default=None,
        help=(
            "Wider EC to MEROPS mapping from map_ec_to_merops.py. When given it "
            "replaces the serum manifest as the set of EC numbers to extract."
        ),
    )
    parser.add_argument(
        "--organism",
        default=TARGET_ORGANISM,
        help="Organism whose protein records are retained.",
    )
    parser.add_argument("--release", default="2026.1", help="Required BRENDA release.")
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Stream the BRENDA snapshot and extract the serum panel kinetic entries.

    :param arguments: Command-line arguments excluding program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)

    # Input validation
    if not args.archive.is_file() or not tarfile.is_tarfile(args.archive):
        print(f"error: BRENDA archive is missing or invalid: {args.archive}", file=sys.stderr)
        return 1
    try:
        if args.ec_mapping is not None:
            manifest_rows = read_ec_mapping(args.ec_mapping)
            scope = f"peptidase mapping {args.ec_mapping.name}"
        else:
            manifest_rows = read_manifest(args.manifest)
            scope = "serum panel manifest"
    except (OSError, ValueError) as error:
        print(f"error: cannot read the EC source: {error}", file=sys.stderr)
        return 1

    ec_index = build_ec_index(manifest_rows)
    if not ec_index:
        print("error: the EC source contains no protease with an EC number.", file=sys.stderr)
        return 1
    print(f"EC numbers requested from BRENDA ({scope}): {len(ec_index)}")

    # Streaming extraction
    ## REMARK: The archive holds one ~677 MB JSON object. It is streamed with ijson
    ## so peak memory stays at the size of the largest single EC record.
    started_at = time.time()
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seen_ec: set[str] = set()
    try:
        for ec_number, record in iter_ec_records(args.archive):
            if ec_number not in ec_index or not isinstance(record, dict):
                continue
            seen_ec.add(ec_number)
            ec_rows, summary = extract_entries(ec_number, record, ec_index[ec_number], args.organism)
            rows.extend(ec_rows)
            summaries.append(summary)
            if len(ec_index) <= 60:
                print(
                    f"  {ec_number}: {len(ec_rows)} {args.organism} entries "
                    f"({summary['n_target_protein_records']}/{summary['n_protein_records']} protein records)"
                )
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"error: cannot stream the BRENDA archive: {error}", file=sys.stderr)
        return 1

    ## Record EC numbers that BRENDA does not contain at all
    missing_ec = sorted(set(ec_index) - seen_ec)
    for ec_number in missing_ec:
        summaries.append(
            {
                "ec_number": ec_number,
                "merops_codes": ";".join(sorted(row["merops_code"] for row in ec_index[ec_number])),
                "n_protein_records": 0,
                "n_target_protein_records": 0,
                "status": "absent_from_brenda",
            }
        )

    # Outputs
    write_table(rows, args.output)
    summary_path = args.output.with_name(args.output.stem + "_coverage.json")
    write_json(
        summary_path,
        {
            "brenda_release": args.release,
            "organism": args.organism,
            "archive": str(args.archive),
            "n_requested_ec": len(ec_index),
            "n_ec_present_in_brenda": len(seen_ec),
            "n_ec_absent_from_brenda": len(missing_ec),
            "ec_absent_from_brenda": missing_ec,
            "n_entries": len(rows),
            "per_ec": sorted(summaries, key=lambda item: item["ec_number"]),
        },
    )
    write_receipt(
        args.output.parent / "ingest_manifest.json",
        "BRENDA",
        args.release,
        f"local parse of {args.archive.name}",
        [
            raw_file_record(args.output, "derived from manual licensed snapshot", "parsed"),
            raw_file_record(summary_path, "derived from manual licensed snapshot", "parsed"),
        ],
    )

    elapsed = time.time() - started_at
    print(
        f"Extracted {len(rows)} BRENDA entries for {len(seen_ec)}/{len(ec_index)} EC numbers "
        f"in {elapsed:.1f}s -> {args.output}"
    )
    if missing_ec:
        print(f"EC numbers absent from BRENDA {args.release}: {', '.join(missing_ec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
