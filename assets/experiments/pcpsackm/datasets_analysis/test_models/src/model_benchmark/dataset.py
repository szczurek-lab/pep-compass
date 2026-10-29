"""Evaluation cohorts for published half-life predictors.

The processed half-life tables are the single source of truth. Duration
classes and the regression-label rule come from the data contract
(``proteolysis_data_manager.integration.half_life.labels``) and are never
redefined here.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

from model_benchmark.config import (
    CANONICAL_ALPHABET,
    COHORTS,
    DATA_MANAGER_SOURCE,
    MAX_LENGTH,
    MIN_LENGTH,
    PRIMARY_COHORT,
    PROCESSED_DATA,
)

logger = logging.getLogger(__name__)

## REMARK: The data manager is a path-based tool, not an installed package;
## notebooks in this project part add its source tree the same way.
if str(DATA_MANAGER_SOURCE) not in sys.path:
    sys.path.insert(0, str(DATA_MANAGER_SOURCE))

from proteolysis_data_manager.integration.half_life.labels import (  # noqa: E402
    CLASS_SCHEME,
    add_duration_classes,
    point_class,
)

MODIFICATION_COLUMNS = (
    "n_terminal_modification",
    "c_terminal_modification",
    "other_modifications",
)


def load_tables():
    """Read the processed half-life tables and attach contract duration classes.

    :return: Experiment table with duration classes and the peptide registry.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    :raises FileNotFoundError: If the processed TSV files are absent.
    """
    experiments_path = PROCESSED_DATA / "half_time_experiments.tsv"
    sequences_path = PROCESSED_DATA / "sequences_half_time.tsv"
    for path in (experiments_path, sequences_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; generate it with the proteolysis data manager first"
            )

    experiments = pd.read_csv(experiments_path, sep="\t")
    sequences = pd.read_csv(sequences_path, sep="\t")
    logger.debug("Read %s experiments and %s sequences", len(experiments), len(sequences))
    return add_duration_classes(experiments), sequences


def _annotate_peptides(frame):
    """Add sequence-derived evaluation flags to a merged experiment table.

    :param pandas.DataFrame frame: Merged experiment and sequence table.
    :return: Table with ``length``, ``canonical`` and ``modification_free`` columns.
    :rtype: pandas.DataFrame
    """
    annotated = frame.copy()
    sequence = annotated.sequence.fillna("")
    annotated["length"] = sequence.str.len()
    annotated["canonical"] = sequence.map(
        lambda text: bool(text) and set(text) <= CANONICAL_ALPHABET
    )

    ## REMARK: A published sequence-only predictor cannot represent chemical
    ## modifications, so peptides carrying them are kept but flagged; the
    ## primary comparison is reported for modification-free peptides as well.
    free_terms = np.logical_and.reduce(
        [annotated[column].fillna("free").eq("free") for column in MODIFICATION_COLUMNS]
    )
    annotated["modification_free"] = (
        free_terms
        & annotated.topology.fillna("linear").eq("linear")
        & annotated.chirality.fillna("l").eq("l")
    )
    return annotated


def _cohort_mask(frame, cohort):
    """Build the boolean selector of one named cohort.

    :param pandas.DataFrame frame: Annotated experiment table.
    :param dict cohort: Cohort definition from :data:`model_benchmark.config.COHORTS`.
    :return: Row selector for the cohort.
    :rtype: pandas.Series
    """
    mask = (
        frame.regression_label_available
        & frame.canonical
        & frame.length.between(MIN_LENGTH, MAX_LENGTH)
    )
    for column in ("species", "matrix", "in_vivo_in_vitro"):
        allowed = cohort[column]
        if allowed is not None:
            mask &= frame[column].isin(allowed)
    return mask


def load_cohort(name=PRIMARY_COHORT, aggregate=True):
    """Return the evaluation cohort used to score published predictors.

    Records are selected by the contract rule ``regression_label_available``
    (exact half-life relation with a positive point estimate), restricted to
    the canonical amino-acid alphabet and to the configured length window.

    :param str name: Cohort key of :data:`model_benchmark.config.COHORTS`.
    :param bool aggregate: Collapse repeated measurements of one peptide into a
        single geometric-mean point estimate. Sequence-only predictors cannot
        distinguish repeated conditions, so aggregation is the default.
    :return: One row per peptide (or per record when ``aggregate`` is false).
    :rtype: pandas.DataFrame
    :raises KeyError: If the cohort name is unknown.
    """
    if name not in COHORTS:
        raise KeyError(f"Unknown cohort {name!r}; available: {sorted(COHORTS)}")

    # Cohort selection
    experiments, sequences = load_tables()
    merged = _annotate_peptides(
        experiments.merge(sequences, on="sequence_id", how="left", validate="many_to_one")
    )
    selected = merged.loc[_cohort_mask(merged, COHORTS[name])].copy()
    selected["log10_half_life"] = np.log10(selected.half_life_seconds)  # (N,)
    logger.info(
        "Cohort %s: %s records, %s peptides", name, len(selected), selected.sequence_id.nunique()
    )
    if not aggregate:
        return selected.reset_index(drop=True)

    # Per-peptide aggregation
    ## REMARK: The geometric mean is the arithmetic mean of log10 half-life,
    ## which is the scale all regression metrics use. The within-peptide
    ## standard deviation is retained as the measurement noise floor against
    ## which model errors must be judged.
    grouped = selected.groupby(["sequence_id", "sequence"], as_index=False).agg(
        log10_half_life=("log10_half_life", "mean"),
        log10_spread=("log10_half_life", "std"),
        record_count=("experiment_id", "size"),
        length=("length", "first"),
        modification_free=("modification_free", "all"),
        sources=("source", lambda values: ";".join(sorted(set(values)))),
        matrices=("matrix", lambda values: ";".join(sorted(set(values)))),
        contexts=("in_vivo_in_vitro", lambda values: ";".join(sorted(set(values.dropna())))),
    )
    grouped["half_life_seconds"] = 10.0 ** grouped.log10_half_life  # (N,)
    grouped["log10_spread"] = grouped.log10_spread.fillna(0.0)

    ## REMARK: The class of an aggregated peptide is recomputed from the
    ## aggregated point with the contract function, so the class scheme stays
    ## identical to the one published for the dataset.
    grouped["duration_class"] = grouped.half_life_seconds.map(point_class)
    grouped["duration_class_scheme"] = CLASS_SCHEME
    grouped["cohort"] = name

    result = grouped.sort_values("sequence_id").reset_index(drop=True)
    logger.info(
        "Cohort %s aggregated to %s peptides (%s modification-free)",
        name,
        len(result),
        int(result.modification_free.sum()),
    )
    return result


def cohort_summary(frame):
    """Summarize one cohort for reporting.

    :param pandas.DataFrame frame: Aggregated cohort from :func:`load_cohort`.
    :return: Single-row summary with counts, class balance and target range.
    :rtype: pandas.DataFrame
    """
    classes = frame.duration_class.value_counts()
    return pd.DataFrame(
        [
            {
                "peptides": len(frame),
                "modification_free": int(frame.modification_free.sum()),
                "records": int(frame.record_count.sum()),
                "median_length": float(frame.length.median()),
                "min_half_life_s": float(frame.half_life_seconds.min()),
                "median_half_life_s": float(frame.half_life_seconds.median()),
                "max_half_life_s": float(frame.half_life_seconds.max()),
                "log10_range": float(
                    frame.log10_half_life.max() - frame.log10_half_life.min()
                ),
                "median_replicate_spread_log10": float(
                    frame.loc[frame.record_count > 1, "log10_spread"].median()
                )
                if (frame.record_count > 1).any()
                else float("nan"),
                **{f"class_{name}": int(classes.get(name, 0)) for name in ("short", "medium", "long")},
            }
        ]
    )
