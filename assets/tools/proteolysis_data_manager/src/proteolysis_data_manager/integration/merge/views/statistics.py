"""Summarize the common numeric columns by source without mixing value categories."""

import pandas as pd

COMMON_FEATURES = (
    "kcat",
    "km",
    "kcat_km",
    "half_life",
    "substrate_depletion_percent",
    "relative_activity_percent",
    "fluorescence_rate_au_s",
    "apparent_rate_s_inv",
    "position_probability_sum",
)


def source_statistics(dataset, frame):
    """Count common feature availability and original source/profile records.

    :param dataset: Common normalized dataset.
    :param frame: Already-computed wide feature table.
    :return: One source per row, with named feature availability columns.
    """
    original = (
        dataset.table("source_records")
        .groupby("source_db")
        .size()
        .rename("source_records")
    )
    result = frame.groupby("source_db", sort=False).agg(
        rows=("row_id", "size"), rows_with_window=("has_window", "sum")
    )
    values = frame.groupby("source_db", sort=False)[list(COMMON_FEATURES)].count()
    result = result.join(values).join(original, how="outer").fillna(0)
    profiles = dataset.table("specificity_profiles").merge(
        dataset.table("source_records")[["record_id", "source_db"]],
        left_on="source_record_id",
        right_on="record_id",
        validate="many_to_one",
    )
    result["profiles"] = (
        profiles.groupby("source_db").size().reindex(result.index, fill_value=0)
    )
    return result.astype("int64").reset_index()


def feature_examples(frame, *, per_source=2):
    """Choose real numerical rows with windows, retaining every source where possible.

    :param frame: Wide common dataset.
    :param int per_source: Maximum representative rows per source.
    :return: Compact preview of common parameter columns and source chemistry.
    """
    selected = frame.loc[frame[list(COMMON_FEATURES)].notna().any(axis=1)].copy()
    weights = pd.Series(
        {
            "kcat": 10,
            "km": 5,
            "kcat_km": 100,
            "half_life": 5,
            "substrate_depletion_percent": 50,
            "relative_activity_percent": 40,
            "fluorescence_rate_au_s": 20,
            "apparent_rate_s_inv": 5,
            "position_probability_sum": 10,
        }
    )
    selected["_preview_rank"] = (
        selected[list(COMMON_FEATURES)].notna().mul(weights).sum(axis=1)
    )
    selected = selected.sort_values(
        ["has_window", "_preview_rank"], ascending=False, kind="stable"
    )
    fields = [
        "source_db",
        "source_dataset",
        "enzyme_name",
        "substrate_name",
        "window",
        *COMMON_FEATURES,
    ]
    examples = []
    for _, group in selected.groupby("source_db", sort=False):
        subsets = []
        remaining = group
        for _ in range(per_source):
            if remaining.empty:
                break
            unseen = remaining.loc[~remaining.source_dataset.isin(subsets)]
            choice = (unseen if not unseen.empty else remaining).iloc[[0]]
            examples.append(choice[fields])
            subsets.append(choice.source_dataset.iloc[0])
            remaining = remaining.drop(index=choice.index)
    return (
        pd.concat(examples, ignore_index=True)
        if examples
        else selected[fields].reset_index(drop=True)
    )
