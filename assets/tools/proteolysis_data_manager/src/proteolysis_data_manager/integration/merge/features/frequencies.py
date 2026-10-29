"""Normalize source cleavage occurrences to common positional probability matrices."""

import numpy as np
import pandas as pd

from proteolysis_data_manager.integration.merge.contract.schema import (
    AA,
    POSITIONS,
    stable_id,
)
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)
PROFILE_COLUMNS = [
    "profile_id",
    "source_db",
    "source_dataset",
    "merops_code",
    "enzyme_name",
    "position",
    "position_index",
    "observed_weight",
    "weighting",
    *AA,
]


def position_probabilities(dataset, *, frame=None):
    """Return common amino-acid probability columns for MEROPS, PICS and qPISA.

    :param dataset: Normalized dataset, preserving source count matrices and windows.
    :param frame: Optional already-computed wide values to avoid decoding them twice.
    :return: One row per source/profile/position with 20 amino-acid probabilities.

    MEROPS and PICS use count/sum(count) among resolved standard residues. PICS
    occurrence rows, enzyme and library strata are preserved. Endpoint qPISA uses
    positive fractional substrate-signal loss as a continuous weight and includes
    only parent-substrate measurements, not product responses or repeated EV6 times.
    Interpretation assumes approximately mixed input; detection and library biases
    remain. Zero positional depth produces null probabilities, not a uniform prior.
    No smoothing, background correction or inferred joint sequence counts are used.
    Time is O(C+8R), space O(C+R), for C original profile cells and R source rows.
    """
    outputs = []
    cells = dataset.table("profile_cells")
    if not cells.empty:
        profiles = dataset.table("specificity_profiles")
        origin = dataset.table("source_records")[
            ["record_id", "source_db", "source_dataset"]
        ]
        meta = profiles.merge(
            origin,
            left_on="source_record_id",
            right_on="record_id",
            validate="many_to_one",
        )
        names = dataset.table("enzymes")[["enzyme_id", "name"]].rename(
            columns={"name": "enzyme_name"}
        )
        meta = meta.merge(names, on="enzyme_id", validate="many_to_one")
        counts = cells.pivot(
            index=["profile_id", "position_index"], columns="residue", values="count"
        ).reindex(columns=list(AA))
        depth = counts.sum(axis=1)
        probabilities = counts.div(depth.replace(0, np.nan), axis=0)
        probabilities["observed_weight"] = depth
        probabilities = probabilities.reset_index().merge(
            meta[
                [
                    "profile_id",
                    "source_db",
                    "source_dataset",
                    "merops_code",
                    "enzyme_name",
                ]
            ],
            on="profile_id",
            validate="many_to_one",
        )
        probabilities["position"] = probabilities.position_index.map(
            dict(enumerate(POSITIONS))
        )
        probabilities["weighting"] = "source_marginal_count"
        outputs.append(probabilities[PROFILE_COLUMNS])
    if frame is None:
        from proteolysis_data_manager.integration.merge.views.wide import common_frame

        frame = common_frame(dataset, include_probabilities=False)
    if frame.empty:
        return (
            pd.concat(outputs, ignore_index=True)
            if outputs
            else pd.DataFrame(columns=PROFILE_COLUMNS)
        )
    # Treat source occurrences and continuous endpoint depletion as different evidence.
    fields = ["source_db", "source_dataset", "merops_code", "enzyme_name", *POSITIONS]
    pics = frame.loc[
        frame.cleavage_occurrence_measurement_id.notna() & frame.has_window
    ].copy()
    pics["_weight"] = pd.to_numeric(pics.cleavage_occurrence)
    pics["_weighting"] = "source_occurrence_count"
    qpisa = frame.loc[
        frame.source_db.eq("qpisa")
        & frame.source_dataset.isin(["EV1", "EV4"])
        & frame.substrate_log2_depletion_measurement_id.notna()
        & frame.has_window
    ].copy()
    qpisa["_weight"] = (qpisa.substrate_depletion_percent / 100).clip(lower=0)
    qpisa["_weighting"] = "positive_depletion_weighted_candidate_windows"
    selected = pd.concat(
        [
            pics[fields + ["_weight", "_weighting"]],
            qpisa[fields + ["_weight", "_weighting"]],
        ],
        ignore_index=True,
    )
    if not selected.empty:
        keys = [
            "source_db",
            "source_dataset",
            "merops_code",
            "enzyme_name",
            "_weighting",
        ]
        for key, group in selected.groupby(keys, sort=False, dropna=False):
            db, subset, code, name, weighting = key
            pid = stable_id("empirical_profile", db, subset, code, name, weighting)
            for i, position in enumerate(POSITIONS):
                valid = group[position].isin(list(AA))
                counts = (
                    group.loc[valid]
                    .groupby(position, sort=False)
                    ._weight.sum()
                    .reindex(list(AA), fill_value=0)
                )
                depth = float(counts.sum())
                values = (
                    (counts / depth).to_dict()
                    if depth > 0
                    else dict.fromkeys(AA, np.nan)
                )
                outputs.append(
                    pd.DataFrame(
                        [
                            {
                                "profile_id": pid,
                                "source_db": db,
                                "source_dataset": subset,
                                "merops_code": code,
                                "enzyme_name": name,
                                "position": position,
                                "position_index": i,
                                "observed_weight": depth,
                                "weighting": weighting,
                                **values,
                            }
                        ]
                    )
                )
    result = (
        pd.concat(outputs, ignore_index=True)
        if outputs
        else pd.DataFrame(columns=PROFILE_COLUMNS)
    )
    logger.info("Computed %s source/profile/position probability rows", len(result))
    return result


def attach_position_probabilities(dataset, frame, *, merops_dataset_index=34):
    """Attach source-specific positional probabilities and their sum to each window.

    :param dataset: Common dataset.
    :param frame: Wide measurement/context frame.
    :param int merops_dataset_index: MEROPS fallback population for explicit enzyme codes.
    :return: Wide frame with p_P4..p_P4p, probability sum and coverage/evidence fields.

    PICS and endpoint qPISA use their own source/library/enzymatic profile. Other
    windows use an explicitly matching MEROPS code. These are descriptive in-sample
    marginal frequencies, not an independently validated probability of cleavage.
    Unsupported residues keep null positional probabilities and explicit coverage.
    """
    profiles = position_probabilities(dataset, frame=frame)
    lookup = {}
    origins = {}
    for row in profiles.itertuples():
        enzyme_key = (
            row.merops_code
            if pd.notna(row.merops_code)
            else "name:" + str(row.enzyme_name)
        )
        key = (row.source_db, str(row.source_dataset), enzyme_key)
        lookup[(key, row.position)] = dict(zip(AA, [getattr(row, aa) for aa in AA]))
        origins[key] = (row.profile_id, row.weighting)
    records = []
    for row in frame[
        [
            "source_db",
            "source_dataset",
            "merops_code",
            "enzyme_name",
            "has_window",
            *POSITIONS,
        ]
    ].itertuples(index=False):
        enzyme_key = (
            row.merops_code
            if pd.notna(row.merops_code)
            else "name:" + str(row.enzyme_name)
        )
        own = (row.source_db, str(row.source_dataset), enzyme_key)
        reference = ("merops", str(merops_dataset_index), row.merops_code)
        key = own if own in origins else reference
        values = {}
        observed = scored = 0
        for position in POSITIONS:
            residue = getattr(row, position)
            if pd.notna(residue):
                observed += 1
            probability = (
                lookup.get((key, position), {}).get(residue, np.nan)
                if pd.notna(residue)
                else np.nan
            )
            if pd.notna(probability):
                scored += 1
            values["p_" + position] = probability
        pid, weighting = origins.get(key, (None, None))
        records.append(
            {
                **values,
                "position_probability_sum": sum(
                    v for v in values.values() if pd.notna(v)
                )
                if scored
                else np.nan,
                "probability_profile_id": pid,
                "probability_source_db": key[0] if pid else None,
                "probability_source_dataset": key[1] if pid else None,
                "probability_weighting": weighting,
                "probability_positions_observed": observed,
                "probability_positions_scored": scored,
                "position_probability_status": "no_window"
                if not row.has_window
                else (
                    "no_profile"
                    if not pid
                    else (
                        "unsupported_or_unobserved_positions"
                        if scored < observed
                        else "complete"
                        if scored == 8
                        else "partial"
                    )
                ),
            }
        )
    return pd.concat(
        [
            frame.reset_index(drop=True),
            pd.DataFrame(records, index=frame.index).reset_index(drop=True),
        ],
        axis=1,
    )
