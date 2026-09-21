#!/usr/bin/env python3
"""Run only the whitelisted serum-calibration acquisition sources."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from _common import acquisition_root, utc_now, write_json
from build_serum_manifest import MANIFEST_COLUMNS


SUPPORTED_SOURCES = (
    "uniprot",
    "brenda",
    "sabio",
    "peplife2_serum",
    "peplife2_plasma",
    "dramp_stability",
    "hpa_blood_ms",
    "hpa_blood_immunoassay",
    "mmp_pics_pmc",
    "mmp_pics_pride",
    "dpp4_qpisa_pmc",
    "dpp4_qpisa_pride",
    "qmsp_ms",
    "adamts13",
    "serine_microarrays_manual",
)


def read_config(path: Path) -> dict[str, Any]:
    """Read the source registry.

    :param path: YAML configuration path.
    :return: Validated mapping with a ``sources`` object.
    :raises ValueError: If the source registry structure is invalid.
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), dict):
        raise ValueError("Source registry must contain a mapping named 'sources'.")
    unexpected = set(payload["sources"]) - set(SUPPORTED_SOURCES)
    if unexpected:
        raise ValueError(f"Source registry contains unsupported source names: {sorted(unexpected)!r}.")
    return payload


def validate_manifest(path: Path) -> None:
    """Validate that a serum manifest exists before source-specific requests.

    :param path: Manifest TSV path.
    :raises ValueError: If the manifest is missing required columns or has no rows.
    """

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or set(MANIFEST_COLUMNS) - set(reader.fieldnames):
            raise ValueError("Manifest does not contain the required serum-protease columns.")
        if not list(reader):
            raise ValueError("Manifest contains no protease rows.")


def source_command(source: str, manifest: Path, dry_run: bool) -> list[str] | None:
    """Build the exact command for one allowed source.

    :param source: Source-registry key.
    :param manifest: Serum protease manifest.
    :param dry_run: Whether the child supports a dry-run switch.
    :return: Command arguments, or ``None`` for manual-only optional data.
    """

    script_dir = Path(__file__).resolve().parent
    python = sys.executable
    commands: dict[str, list[str] | None] = {
        "uniprot": [python, str(script_dir / "map_uniprot_serum_proteases.py"), "--manifest", str(manifest)],
        "brenda": [python, str(script_dir / "check_brenda_snapshot.py")],
        "sabio": [python, str(script_dir / "download_sabio_serum_panel.py"), "--manifest", str(manifest), "--continue-on-error"],
        "peplife2_serum": [python, str(script_dir / "download_peplife2.py"), "human_serum"],
        "peplife2_plasma": [python, str(script_dir / "download_peplife2.py"), "human_plasma"],
        "dramp_stability": [python, str(script_dir / "download_fixed_sources.py"), "dramp_stability"],
        "hpa_blood_ms": [python, str(script_dir / "download_fixed_sources.py"), "hpa_blood_ms"],
        "hpa_blood_immunoassay": [python, str(script_dir / "download_fixed_sources.py"), "hpa_blood_immunoassay"],
        "mmp_pics_pmc": [python, str(script_dir / "download_pmc_supplements.py"), "mmp_pics"],
        "mmp_pics_pride": [python, str(script_dir / "download_pride_project.py"), "PXD002265", "--profile", "mmp_pics"],
        "dpp4_qpisa_pmc": [python, str(script_dir / "download_pmc_supplements.py"), "dpp4_qpisa"],
        "dpp4_qpisa_pride": [python, str(script_dir / "download_pride_project.py"), "PXD042089", "--profile", "dpp4_qpisa"],
        "qmsp_ms": [python, str(script_dir / "download_pmc_supplements.py"), "qmsp_ms"],
        "adamts13": [python, str(script_dir / "download_pmc_supplements.py"), "adamts13"],
        "serine_microarrays_manual": None,
    }
    command = commands[source]
    if command is not None and dry_run and source == "sabio":
        command.append("--dry-run")
    return command


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=acquisition_root() / "manifests" / "serum_proteases.tsv",
        help="Prebuilt serum-protease manifest TSV.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "serum_sources.yaml",
        help="Serum-source registry YAML (defaults to the registry beside this script).",
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=SUPPORTED_SOURCES,
        help="Run only one named source. Repeat to select a set.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan SABIO queries without sending those requests.")
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Run the configured acquisition sources in dependency order.

    :param arguments: Command-line arguments excluding program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)
    try:
        validate_manifest(args.manifest)
        config = read_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: acquisition setup is invalid: {error}", file=sys.stderr)
        return 2

    configured_sources = list(config["sources"])
    selected_sources = args.source if args.source else configured_sources
    results: list[dict[str, object]] = []
    required_failure = False
    for source in selected_sources:
        settings = config["sources"].get(source)
        if not isinstance(settings, dict):
            print(f"error: source {source!r} is not configured.", file=sys.stderr)
            return 2
        command = source_command(source, args.manifest, args.dry_run)
        if command is None:
            results.append({"source": source, "status": "optional_missing", "command": None})
            continue
        print(f"Running source: {source}")
        completed = subprocess.run(command, check=False)
        status = "complete" if completed.returncode == 0 else "failed"
        results.append({"source": source, "status": status, "returncode": completed.returncode, "command": command})
        if completed.returncode != 0 and bool(settings.get("required", False)):
            required_failure = True

    report_path = acquisition_root() / "raw" / "acquisition_run_manifest.json"
    write_json(
        report_path,
        {
            "created_at": utc_now(),
            "manifest": str(args.manifest),
            "config": str(args.config),
            "results": results,
        },
    )
    if required_failure:
        print(f"error: one or more required sources failed. Report: {report_path}", file=sys.stderr)
        return 1
    print(f"Completed configured source acquisition. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
