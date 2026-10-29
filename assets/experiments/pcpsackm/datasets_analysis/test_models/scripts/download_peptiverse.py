"""Download the published PeptiVerse half-life artifacts and their backbones.

Only the half-life property is fetched, together with the three representation
backbones its models depend on. Nothing is trained here.

Usage::

    .venv/bin/python scripts/download_peptiverse.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

MODULE_SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(MODULE_SOURCE) not in sys.path:
    sys.path.insert(0, str(MODULE_SOURCE))

from model_benchmark.config import PEPTIVERSE_DIR  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_REPOSITORY = "ChatterjeeLab/PeptiVerse"
## REMARK: The half-life models and the files needed to interpret them. The
## adapter re-implements batched inference, so the upstream inference.py is
## downloaded for reference and verification only, never imported.
ALLOW_PATTERNS = (
    "README.md",
    "basic_models.txt",
    "best_models.txt",
    "requirements.txt",
    "inference.py",
    "tokenizer/*",
    "training_classifiers/half_life/**",
    "training_data/half_life_smiles.csv",
)
BACKBONE_REPOSITORIES = (
    "facebook/esm2_t33_650M_UR50D",
    "aaronfeller/PeptideCLM-23M-all",
    "DeepChem/ChemBERTa-77M-MLM",
)
BACKBONE_ALLOW_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "*.bin",
    "*.safetensors",
)


def download_models(target):
    """Fetch the half-life artifacts of the PeptiVerse model repository.

    :param pathlib.Path target: Local directory receiving the snapshot.
    :return: Path of the downloaded snapshot.
    :rtype: pathlib.Path
    """
    from huggingface_hub import snapshot_download

    logger.info("Downloading %s half-life artifacts into %s", MODEL_REPOSITORY, target)
    path = snapshot_download(
        repo_id=MODEL_REPOSITORY,
        repo_type="model",
        local_dir=str(target),
        allow_patterns=list(ALLOW_PATTERNS),
    )
    return Path(path)


def download_backbones():
    """Fetch the representation backbones used by the half-life models.

    :return: Mapping of repository identifier to local snapshot path.
    :rtype: dict[str, str]
    """
    from huggingface_hub import snapshot_download

    resolved = {}
    for repository in BACKBONE_REPOSITORIES:
        logger.info("Downloading backbone %s", repository)
        resolved[repository] = snapshot_download(
            repo_id=repository,
            allow_patterns=list(BACKBONE_ALLOW_PATTERNS),
        )
    return resolved


def summarize(snapshot):
    """Report which half-life model directories arrived and how large they are.

    :param pathlib.Path snapshot: Downloaded snapshot directory.
    :return: One record per half-life model directory.
    :rtype: list[dict]
    """
    root = snapshot / "training_classifiers/half_life"
    records = []
    for directory in sorted(path for path in root.glob("*") if path.is_dir()):
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        records.append(
            {
                "model": directory.name,
                "files": [path.name for path in files],
                "megabytes": round(sum(path.stat().st_size for path in files) / 1e6, 2),
            }
        )
    return records


def main():
    """Run the download and write a receipt next to the artifacts.

    :rtype: None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-backbones",
        action="store_true",
        help="Download only the half-life heads, not the representation backbones",
    )
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    PEPTIVERSE_DIR.mkdir(parents=True, exist_ok=True)

    # Published half-life heads
    snapshot = download_models(PEPTIVERSE_DIR)
    models = summarize(snapshot)
    for record in models:
        logger.info("Model %s: %s MB", record["model"], record["megabytes"])

    # Representation backbones
    backbones = {}
    if not arguments.skip_backbones:
        backbones = download_backbones()

    receipt = {
        "model_repository": MODEL_REPOSITORY,
        "snapshot": str(snapshot),
        "half_life_models": models,
        "backbones": backbones,
    }
    receipt_path = PEPTIVERSE_DIR / "download_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", receipt_path)


if __name__ == "__main__":
    main()
