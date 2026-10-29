"""Overlap between our evaluation cohorts and published training sets.

A published predictor evaluated on peptides it was trained on reports its
training fit, not its transfer. The PeptiVerse repository ships the half-life
training table as SMILES, so the overlap can be determined exactly by matching
canonical structures rather than guessing from the data sources.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from model_benchmark.config import PEPTIVERSE_DIR

logger = logging.getLogger(__name__)

TRAINING_TABLE = "training_data/half_life_smiles.csv"
SECONDS_PER_HOUR = 3600.0


def _inchikey(smiles):
    """Return the InChIKey of a SMILES string.

    :param str smiles: SMILES string.
    :return: InChIKey, or None when RDKit rejects the structure.
    :rtype: str | None
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToInchiKey(molecule)


def peptiverse_training_table(root=PEPTIVERSE_DIR):
    """Read the published half-life training table with structure keys.

    :param pathlib.Path root: Downloaded PeptiVerse snapshot.
    :return: Training rows with an InChIKey column.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If the training table is absent from the snapshot.
    """
    path = root / TRAINING_TABLE
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run scripts/download_peptiverse.py")

    table = pd.read_csv(path)
    table["inchikey"] = table.SMILES.map(_inchikey)
    ## REMARK: The published column named half_life holds seconds, and
    ## half_life_hours is that value divided by 3600; the hours column is used
    ## so the comparison stays in the authors' own unit.
    seconds = table.half_life_hours * SECONDS_PER_HOUR  # (N,)
    table["training_log10_seconds"] = np.where(seconds > 0, np.log10(seconds.clip(lower=1e-12)), np.nan)
    logger.info("Published half-life training set: %s rows", len(table))
    return table


def cohort_structure_keys(cohort):
    """Attach canonical structure keys to an evaluation cohort.

    :param pandas.DataFrame cohort: Aggregated cohort with a ``sequence`` column.
    :return: Cohort with ``smiles`` and ``inchikey`` columns.
    :rtype: pandas.DataFrame
    """
    from model_benchmark.predictors.peptiverse import sequence_to_smiles

    annotated = cohort.copy()
    annotated["smiles"] = annotated.sequence.map(sequence_to_smiles)
    annotated["inchikey"] = annotated.smiles.map(_inchikey)
    return annotated


def peptiverse_overlap(cohort, root=PEPTIVERSE_DIR):
    """Identify cohort peptides present in the published training set.

    :param pandas.DataFrame cohort: Aggregated cohort from the dataset module.
    :param pathlib.Path root: Downloaded PeptiVerse snapshot.
    :return: Matched peptides and a summary of the overlap.
    :rtype: tuple[pandas.DataFrame, dict]
    """
    training = peptiverse_training_table(root)
    annotated = cohort_structure_keys(cohort)

    ## REMARK: One structure can appear in several published training rows, so
    ## duplicated keys are collapsed to their median target before matching.
    training_keys = (
        training.dropna(subset=["inchikey"])
        .groupby("inchikey")
        .agg(half_life=("half_life", "median"), training_rows=("half_life", "size"))
    )
    matched = annotated.loc[annotated.inchikey.isin(training_keys.index)].copy()
    if not matched.empty:
        matched["training_half_life_seconds"] = matched.inchikey.map(training_keys.half_life)
        matched["training_rows"] = matched.inchikey.map(training_keys.training_rows)
        matched["training_log10_seconds"] = np.log10(matched.training_half_life_seconds)
        matched["cohort_minus_training_log10"] = (
            matched.log10_half_life - matched.training_log10_seconds
        )

    summary = {
        "cohort_peptides": int(len(annotated)),
        "training_rows": int(len(training)),
        "training_structures": int(training.inchikey.nunique()),
        "overlapping_peptides": int(len(matched)),
        "overlap_share": float(len(matched) / len(annotated)) if len(annotated) else 0.0,
        "median_target_difference_log10": float(
            matched.cohort_minus_training_log10.abs().median()
        )
        if not matched.empty
        else float("nan"),
    }
    logger.info(
        "PeptiVerse training overlap: %s of %s cohort peptides (%.1f%%)",
        summary["overlapping_peptides"],
        summary["cohort_peptides"],
        100 * summary["overlap_share"],
    )
    return matched, summary


def split_by_overlap(cohort, overlap):
    """Split a cohort into peptides seen and unseen by the published training set.

    :param pandas.DataFrame cohort: Aggregated cohort from the dataset module.
    :param pandas.DataFrame overlap: Matched peptides from :func:`peptiverse_overlap`.
    :return: Held-out and training-overlapping parts of the cohort.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    seen_ids = set(overlap.sequence_id) if not overlap.empty else set()
    unseen = cohort.loc[~cohort.sequence_id.isin(seen_ids)].reset_index(drop=True)
    seen = cohort.loc[cohort.sequence_id.isin(seen_ids)].reset_index(drop=True)
    return unseen, seen
