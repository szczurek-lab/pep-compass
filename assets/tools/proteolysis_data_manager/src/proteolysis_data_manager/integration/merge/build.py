"""Compose explicitly selected dataset parsers and write the common SQLite bundle."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from proteolysis_data_manager.integration.merge.contract.builder import ContractBuilder
from proteolysis_data_manager.integration.merge.features.merops import add_merops_scores
from proteolysis_data_manager.integration.merge.registry import (
    SOURCES,
    parse_dataset,
    verify_source,
)
from proteolysis_data_manager.integration.merge.storage.sqlite import write_dataset
from proteolysis_data_manager.utils.io import sha256
from proteolysis_data_manager.utils.logger import get_custom_logger
from proteolysis_data_manager.utils.paths import repository_root

logger = get_custom_logger(__name__)


def build_dataset(
    data_root, *, sources=None, score_windows=False, merops_dataset_index=34
):
    """Parse selected acquired sources, validate composition and optionally score windows.

    :param data_root: Parent directory containing acquired series roots.
    :param sources: Explicit source names; defaults to the full registered set.
    :param bool score_windows: Compute source-defined MEROPS probability features.
    :param int merops_dataset_index: Explicit source population used for scoring.
    :return: Validated common dataset in memory; no files are written here.
    """
    data_root = Path(data_root).resolve()
    sources = tuple(SOURCES) if sources is None else tuple(sources)
    if len(set(sources)) != len(sources):
        raise ValueError("Repeated dataset selection")
    if unknown := set(sources) - set(SOURCES):
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    b = ContractBuilder(data_root)
    receipts = {}
    verified = set()
    for name in sources:
        logger.info("Parsing source %s into the common contract", name)
        acquisition = (
            "dpp4_qpisa"
            if name.startswith("qpisa_")
            else ("mmp_pics" if name == "mmp_pics_qf" else name)
        )
        if acquisition not in verified:
            verify_source(name, data_root)
            verified.add(acquisition)
        dataset = parse_dataset(name, data_root, verify=False)
        for table, frame in dataset.tables.items():
            for record in frame.to_dict("records"):
                b.add(table, **record)
        manifest = data_root / SOURCES[name] / "raw/manifest.json"
        receipts[str(manifest)] = sha256(manifest)
    result = b.finish()
    result.metadata = {
        "sources": list(sources),
        "built_at_utc": datetime.now(UTC).isoformat(),
        "raw_manifest_sha256": receipts,
        "condition_policy": "Equal recorded descriptions share IDs; assays never merge on conditions alone.",
        "identity_policy": "Source identities retained; numerical/sequence coincidence never deduplicates measurements.",
        "merops_score": "sum of original positional probabilities; no log-odds",
        "implementation_sha256": {
            str(path.relative_to(Path(__file__).parent)): sha256(path)
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        },
    }
    if score_windows:
        if "merops" not in verified:
            verify_source("merops", data_root)
            manifest = data_root / SOURCES["merops"] / "raw/manifest.json"
            receipts[str(manifest)] = sha256(manifest)
        result = add_merops_scores(
            result,
            data_root / "window_neighbourhood_datasets/raw/merops",
            dataset_index=merops_dataset_index,
        )
        result.metadata["scoring_merops_dataset_index"] = merops_dataset_index
    return result


def main(argv=None):
    """Build the selected sources into one derived SQLite database.

    :param argv: Optional command-line arguments.
    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=repository_root() / "data/proteolysis"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", action="append", choices=tuple(SOURCES))
    parser.add_argument("--score-windows", action="store_true")
    parser.add_argument("--merops-dataset-index", type=int, default=34)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    dataset = build_dataset(
        args.data_root,
        sources=args.source,
        score_windows=args.score_windows,
        merops_dataset_index=args.merops_dataset_index,
    )
    path = write_dataset(
        dataset,
        args.output or args.data_root / "merged_datasets/proteolysis.sqlite",
        replace=args.replace,
    )
    logger.info("Wrote unified dataset to %s", path)
    print(dataset.summary().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
