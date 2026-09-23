#!/usr/bin/env python3
"""Download the human reference proteome and compute its amino-acid composition.

The composition is the reference distribution used as the null of the MEROPS
specificity diagnostics. It is acquired once and stored as a small JSON file, so
that analyses read a fixed, auditable number instead of reparsing the proteome or
hard-coding frequencies from the literature.

Only the reviewed (Swiss-Prot) canonical sequences of UniProt proteome
UP000005640 are requested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlencode

from _common import download_url, raw_file_record, repository_root, write_json, write_receipt


UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
PROTEOME_ACCESSION = "UP000005640"
QUERY = f"(proteome:{PROTEOME_ACCESSION}) AND (reviewed:true)"
DEFAULT_TIMEOUT_SECONDS = 600


def default_destination() -> Path:
    """Return the directory holding the reference proteome artifacts.

    :return: Absolute directory path.
    """

    return repository_root() / "data" / "reference" / "human_proteome"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    :return: Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=default_destination(),
        help="Directory for the FASTA snapshot, its composition and the receipt.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Socket timeout for the UniProt stream request.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when the FASTA snapshot is already present.",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Download the proteome and write its residue composition.

    :param arguments: Command-line arguments excluding program name.
    :return: Process exit status.
    """

    args = build_parser().parse_args(arguments)
    fasta_path = args.destination / f"{PROTEOME_ACCESSION}_reviewed.fasta.gz"
    composition_path = args.destination / "composition.json"

    # Acquisition
    ## The query contains spaces and parentheses, so it is percent-encoded rather
    ## than interpolated into the URL.
    url = f"{UNIPROT_STREAM_URL}?" + urlencode(
        {"query": QUERY, "format": "fasta", "compressed": "true"}
    )
    if fasta_path.is_file() and not args.force:
        print(f"FASTA snapshot already present: {fasta_path}")
    else:
        print(f"Requesting {QUERY} from UniProt ...")
        try:
            download_url(url, fasta_path, timeout_seconds=args.timeout_seconds, accept="text/plain")
        except Exception as error:  # noqa: BLE001 - the CLI reports any acquisition failure
            print(f"error: cannot download the human reference proteome: {error}", file=sys.stderr)
            return 1

    # Composition
    ## REMARK: The parser is imported from the library rather than reimplemented,
    ## so the stored frequencies and any frequency computed during an analysis
    ## come from one definition of the alphabet and of the exclusion rule.
    sys.path.insert(0, str(repository_root() / "src"))
    from pep_compass.optimization.components.helpers.proteolysis.background import (  # noqa: E402
        amino_acid_background_from_fasta,
    )
    from pep_compass.optimization.components.helpers.proteolysis.cleavage import (  # noqa: E402
        MEROPS_AMINO_ACIDS,
    )

    try:
        frequencies, counts = amino_acid_background_from_fasta(fasta_path)
    except (OSError, ValueError) as error:
        print(f"error: cannot compute the residue composition: {error}", file=sys.stderr)
        return 1

    write_json(
        composition_path,
        {
            "source_database": "UniProt",
            "proteome_accession": PROTEOME_ACCESSION,
            "query": QUERY,
            "request_url": url,
            "alphabet": MEROPS_AMINO_ACIDS,
            "n_sequences": counts["n_sequences"],
            "n_residues_in_alphabet": sum(counts[residue] for residue in MEROPS_AMINO_ACIDS),
            "n_residues_outside_alphabet": counts["non_standard"],
            "counts": {residue: counts[residue] for residue in MEROPS_AMINO_ACIDS},
            "frequencies": {
                residue: float(value) for residue, value in zip(MEROPS_AMINO_ACIDS, frequencies)
            },
        },
    )
    write_receipt(
        args.destination / "download_manifest.json",
        "UniProt",
        PROTEOME_ACCESSION,
        url,
        [
            raw_file_record(fasta_path, url, "downloaded"),
            raw_file_record(composition_path, "derived from the FASTA snapshot", "computed"),
        ],
    )

    print(f"Sequences: {counts['n_sequences']:,}")
    print(f"Residues in the alphabet: {sum(counts[r] for r in MEROPS_AMINO_ACIDS):,}")
    print(f"Residues outside the alphabet: {counts['non_standard']:,}")
    ranked = sorted(zip(MEROPS_AMINO_ACIDS, frequencies), key=lambda item: -item[1])
    print("Most and least frequent residues:")
    for residue, value in ranked[:3] + ranked[-3:]:
        print(f"  {residue}  {value:.4f}")
    print(f"Written: {composition_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
