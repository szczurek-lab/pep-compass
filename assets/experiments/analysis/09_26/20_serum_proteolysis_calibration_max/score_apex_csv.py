"""Batch-score a peptide CSV with a registered APEX ensemble."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import yaml

from pep_compass.optimization.components.oracles.strategies.apex.predictor import (
    APEXPredictor,
)


logger = logging.getLogger(__name__)


def _load_settings(path: Path) -> dict[str, Any]:
    """Load and validate the batch-scoring YAML configuration."""
    with path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError("The YAML root must be a mapping.")
    required = ("input_csv", "output_csv", "sequence_column", "model")
    missing = [key for key in required if not settings.get(key)]
    if missing:
        raise ValueError(f"Missing required settings: {', '.join(missing)}")
    return settings


def _resolve_path(value: str, repository_root: Path) -> Path:
    """Resolve a configuration path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def run(settings_path: Path, repository_root: Path) -> Path:
    """Score all input sequences and write one row per input row.

    :param settings_path: YAML configuration path.
    :param repository_root: Repository root used for relative paths.
    :return: Path to the written output CSV.
    """
    settings = _load_settings(settings_path)
    input_path = _resolve_path(str(settings["input_csv"]), repository_root)
    output_path = _resolve_path(str(settings["output_csv"]), repository_root)
    sequence_column = str(settings["sequence_column"])
    batch_size = int(settings.get("batch_size", 3000))
    device = str(settings.get("device", "cpu"))
    model = str(settings["model"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    logger.info("Loading APEX ensemble model=%s on device=%s", model, device)
    predictor = APEXPredictor(
        device=device,
        model=model,
        batch_size=batch_size,
    )
    logger.info("Resolved %d pathogen outputs", len(predictor.pathogen_list))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(newline="", encoding="utf-8") as input_stream:
        reader = csv.DictReader(input_stream)
        if reader.fieldnames is None or sequence_column not in reader.fieldnames:
            raise ValueError(f"Input CSV requires column: {sequence_column}")
        input_fields = list(reader.fieldnames)
        output_fields = input_fields + list(predictor.pathogen_list)
        with output_path.open("w", newline="", encoding="utf-8") as output_stream:
            writer = csv.DictWriter(output_stream, fieldnames=output_fields)
            writer.writeheader()
            batch: list[dict[str, str]] = []
            processed = 0

            # Batch inference
            ## Preserve the original input rows and append APEX MIC predictions.
            def flush() -> None:
                nonlocal processed
                if not batch:
                    return
                sequences = [row[sequence_column].strip() for row in batch]
                if any(not sequence for sequence in sequences):
                    raise ValueError("Input contains an empty peptide sequence.")
                predictions = predictor.predict(sequences)
                for row, values in zip(batch, predictions, strict=True):
                    result = dict(row)
                    result.update(
                        {
                            pathogen: f"{float(value):.8g}"
                            for pathogen, value in zip(
                                predictor.pathogen_list, values, strict=True
                            )
                        }
                    )
                    writer.writerow(result)
                processed += len(batch)
                logger.info("Scored %d sequences", processed)
                batch.clear()

            for row in reader:
                batch.append(row)
                if len(batch) >= batch_size:
                    flush()
            flush()

    logger.info("Wrote APEX predictions to %s", output_path)
    return output_path


def main() -> int:
    """Run the configured batch evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run(args.config, args.repository_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
