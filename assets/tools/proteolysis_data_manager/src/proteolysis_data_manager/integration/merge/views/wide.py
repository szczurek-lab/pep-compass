"""Construct the common feature columns from explicitly associated measurements."""

import numpy as np
import pandas as pd

from proteolysis_data_manager.integration.merge.contract.features import FEATURE_UNITS
from proteolysis_data_manager.integration.merge.contract.schema import (
    POSITIONS,
    QUANTITIES,
    stable_id,
)
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)
MEASURED = tuple(q for q in QUANTITIES if not q.startswith("merops_"))
GROUP_KEYS = [
    "source_db",
    "source_dataset",
    "assay_id",
    "subject_participant_id",
    "context_id",
    "time_s",
    "replicate_id",
    "measurement_scope",
]
DETAILS = [
    "value",
    "unit",
    "relation",
    "lower_bound",
    "upper_bound",
    "measurement_id",
    "uncertainty_kind",
    "uncertainty_value",
]


def measurement_rows(dataset, *, source=None):
    """Join actual parameter values, chemistry and cut windows through source keys.

    :param dataset: Common normalized dataset.
    :param source: Optional source database selector.
    :return: One row per measurement/context association, including unlinked values.
    """
    rows = dataset.measurements(source=source)
    if rows.empty:
        return rows
    links = dataset.table("observation_contexts")
    rows = rows.merge(links, on="measurement_id", how="left", validate="one_to_many")
    contexts = dataset.contexts()
    fields = [
        "context_id",
        "substrate_participant_id",
        "enzyme_participant_id",
        "cut_after_residue",
        "coordinate_basis",
        "site_evidence",
        *POSITIONS,
        *[p + "_status" for p in POSITIONS],
    ]
    rows = rows.merge(
        contexts[fields], on="context_id", how="left", validate="many_to_one"
    )
    participants = dataset.table("participants")
    # Use a unique enzyme in the source assay only when no explicit site enzyme exists.
    enzymes = participants.loc[participants.enzyme_id.notna()]
    unique = (
        enzymes.loc[~enzymes.assay_id.duplicated(keep=False)]
        .set_index("assay_id")
        .participant_id
    )
    rows["enzyme_participant_id"] = rows.enzyme_participant_id.where(
        rows.enzyme_participant_id.notna(), rows.assay_id.map(unique)
    ).astype("string")
    rows = rows.merge(
        participants[["participant_id", "enzyme_id"]],
        left_on="enzyme_participant_id",
        right_on="participant_id",
        how="left",
        validate="many_to_one",
    ).drop(columns="participant_id")
    rows = rows.merge(
        dataset.table("enzymes")[
            ["enzyme_id", "name", "merops_code", "ec_number", "organism"]
        ].rename(columns={"name": "enzyme_name", "organism": "enzyme_organism"}),
        on="enzyme_id",
        how="left",
        validate="many_to_one",
    )
    entity = participants.set_index("participant_id").substrate_id
    parent = rows.substrate_participant_id.where(
        rows.substrate_participant_id.notna(), rows.subject_participant_id
    )
    rows["substrate_id"] = parent.map(entity)
    rows = rows.merge(
        dataset.table("substrates")[
            ["substrate_id", "name", "sequence", "chemistry_json"]
        ].rename(columns={"name": "substrate_name", "sequence": "substrate_sequence"}),
        on="substrate_id",
        how="left",
        validate="many_to_one",
    )
    rows["has_window"] = rows.context_id.notna()

    def window(record):
        if not record.has_window:
            return None
        tokens = [
            str(getattr(record, p)) if pd.notna(getattr(record, p)) else "-"
            for p in POSITIONS
        ]
        return " ".join(tokens[:4]) + " | " + " ".join(tokens[4:])

    rows["window"] = [
        window(r) for r in rows[["has_window", *POSITIONS]].itertuples(index=False)
    ]
    return rows


def common_frame(
    dataset,
    *,
    source=None,
    quantities=None,
    merops_dataset_index=34,
    include_probabilities=True,
):
    """Return named numerical feature columns, retaining all original associations.

    :param dataset: Common dataset or SQLite reader.
    :param source: Optional source database selector.
    :param quantities: Optional measurement-name row selector; grouping is unchanged.
    :param int merops_dataset_index: Explicit MEROPS feature population selector.
    :return: Wide frame with source, substrate, enzyme, window and numeric columns.

    Parameters share a row only within the same source assay, measured participant,
    context, time, replicate and scope. If a quantity repeats within that group,
    every measurement in that group stays unpaired rather than being averaged.
    Time/space are O((N+L) log(N+L)) and O((N+L)Q), respectively, where N denotes
    measurements, L site links and Q the fixed number of common feature columns.
    """
    logger.info("Constructing common numeric feature columns")
    rows = measurement_rows(dataset, source=source)
    if rows.empty:
        return pd.DataFrame(
            columns=list(
                dict.fromkeys(
                    [
                        "row_id",
                        *GROUP_KEYS,
                        "enzyme_name",
                        "substrate_name",
                        "window",
                        "has_window",
                        *POSITIONS,
                        *MEASURED,
                        *FEATURE_UNITS,
                    ]
                )
            )
        )
    # Detect ambiguous groups before any quantity selection or pivot operation.
    base = rows.groupby(GROUP_KEYS, dropna=False, sort=False).ngroup()
    repeats = (
        rows.assign(_group=base)
        .groupby(["_group", "quantity"], sort=False)
        .measurement_id.transform("size")
        .gt(1)
    )
    ambiguous = set(base.loc[repeats])
    ambiguous.update(base.loc[rows.subject_participant_id.isna()])
    rows["parameter_pairing"] = np.where(
        base.isin(ambiguous), "multiple_values_unpaired", "unique_source_group"
    )
    group_keys = {
        r[0]: stable_id("feature_row", list(r[1:]))
        for r in rows[GROUP_KEYS]
        .groupby(base, sort=False)
        .first()
        .itertuples(name=None)
    }
    rows["row_id"] = [
        stable_id("feature_row", group_keys[g], mid)
        if g in ambiguous
        else group_keys[g]
        for g, mid in zip(base, rows.measurement_id)
    ]
    # Preserve scalar metadata while placing each actual parameter in its column.
    metadata = [
        c
        for c in rows
        if c
        not in {
            "quantity",
            *DETAILS,
            "raw_value",
            "raw_unit",
            "derivation",
            "source_record_id",
            "record_id",
            "attribution",
            "equivalence_group",
            "ci_lower",
            "ci_upper",
            "ci_level",
            "replicate_count",
        }
    ]
    frame = rows.drop_duplicates("row_id")[metadata].set_index("row_id")
    columns = {}
    for detail in DETAILS:
        wide = rows.pivot(index="row_id", columns="quantity", values=detail)
        for quantity in MEASURED:
            column = quantity if detail == "value" else quantity + "_" + detail
            columns[column] = (
                wide[quantity]
                if quantity in wide
                else pd.Series(
                    np.nan
                    if detail
                    in {"value", "lower_bound", "upper_bound", "uncertainty_value"}
                    else None,
                    index=frame.index,
                )
            )
    # Keep the exact source-record and attribution of each scalar after aggregation.
    for detail in ["source_record_id", "derivation", "attribution"]:
        wide = rows.pivot(index="row_id", columns="quantity", values=detail)
        for quantity in MEASURED:
            columns[quantity + "_" + detail] = (
                wide[quantity]
                if quantity in wide
                else pd.Series(None, index=frame.index, dtype=object)
            )
    frame = pd.concat(
        [frame, pd.DataFrame(columns, index=frame.index)], axis=1
    ).reset_index()
    for quantity in MEASURED:
        frame[quantity] = pd.to_numeric(frame[quantity], errors="coerce")
        for suffix in ["unit", "relation", "measurement_id", "uncertainty_kind"]:
            frame[quantity + "_" + suffix] = frame[quantity + "_" + suffix].astype(
                object
            )
    from proteolysis_data_manager.integration.merge.features.activity import (
        transform_activity,
    )

    frame = transform_activity(dataset, frame)
    # Attach three MEROPS quantities as columns rather than rows or category counts.
    features = dataset.table("context_features")
    selected = features.loc[
        features.method.eq(
            f"original_merops_dataset_{merops_dataset_index}_available_positions_v1"
        )
    ]
    primary = selected.loc[
        selected.quantity.eq("merops_probability_sum"),
        [
            "context_id",
            "profile_id",
            "value",
            "status",
            "n_observed_positions",
            "n_scored_positions",
        ],
    ].rename(columns={"value": "merops_probability_sum", "status": "merops_status"})
    for quantity in ["merops_log_probability_sum", "merops_probability_product"]:
        other = selected.loc[
            selected.quantity.eq(quantity), ["context_id", "value"]
        ].rename(columns={"value": quantity})
        primary = primary.merge(
            other, on="context_id", how="left", validate="one_to_one"
        )
    frame = frame.merge(primary, on="context_id", how="left", validate="many_to_one")
    frame["merops_status"] = frame.merops_status.fillna(
        pd.Series(
            np.where(frame.has_window, "not_computed", "no_context"), index=frame.index
        )
    )
    frame["merops_dataset_index"] = merops_dataset_index
    if include_probabilities:
        from proteolysis_data_manager.integration.merge.features.frequencies import (
            attach_position_probabilities,
        )

        frame = attach_position_probabilities(
            dataset, frame, merops_dataset_index=merops_dataset_index
        )
    if quantities is not None:
        unknown = set(quantities) - set(MEASURED)
        if unknown:
            raise ValueError(f"Unknown measurement quantities: {sorted(unknown)}")
        selected = np.zeros(len(frame), dtype=bool)
        for q in quantities:
            selected |= frame[q + "_measurement_id"].notna().to_numpy()
        if "kcat_km" in quantities:
            selected |= frame.kcat_km.notna().to_numpy()
        frame = frame.loc[selected]
    return frame.reset_index(drop=True)
