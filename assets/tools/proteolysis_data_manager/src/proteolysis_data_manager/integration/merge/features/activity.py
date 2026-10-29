"""Transform reported responses into common percentages and explicit rate features."""

import json

import numpy as np
import pandas as pd

from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def transform_activity(dataset, frame):
    """Compute common numeric responses while retaining their measurement basis.

    :param dataset: Normalized source tables, including original source inputs.
    :param frame: Wide measurement/context table.
    :return: New frame with percentages, efficiency, apparent rates and QF slopes.

    Depletion is relative to the matched control signal, not an absolute molecular
    conversion calibration. Endpoint and half-life rates assume exponential loss.
    A fluorescence slope is fitted on the source-selected interval; no fully
    cleaved fluorescence calibration or intrinsic kcat is invented.
    """
    out = frame.copy()
    # Combine only uniquely paired kinetic constants from the same source group.
    supplied = out.kcat_km_measurement_id.notna()
    pair = (
        (~supplied)
        & out.subject_participant_id.notna()
        & out.kcat.notna()
        & out.km.gt(0)
        & out.parameter_pairing.eq("unique_source_group")
    )
    out["kcat_km_basis"] = np.where(supplied, "reported", None)
    if pair.any():
        out.loc[pair, "kcat_km"] = out.loc[pair, "kcat"] / out.loc[pair, "km"]
        out.loc[pair, "kcat_km_unit"] = "M^-1 s^-1"
        out.loc[pair, "kcat_km_relation"] = np.where(
            out.loc[pair, "kcat_relation"].eq("approx")
            | out.loc[pair, "km_relation"].eq("approx"),
            "approx",
            "eq",
        )
        out.loc[pair, "kcat_km_basis"] = "kcat_divided_by_km_same_source_group"
    # A log2 control/treated difference has an exact relative signal-ratio inverse.
    depleted = out.substrate_log2_depletion.notna()
    with np.errstate(over="ignore", invalid="ignore"):
        remaining = np.exp2(
            -pd.to_numeric(out.substrate_log2_depletion, errors="coerce")
        )
    out.loc[depleted, "remaining_fraction"] = remaining.loc[depleted]
    out["substrate_remaining_percent"] = 100 * pd.to_numeric(
        out.remaining_fraction, errors="coerce"
    )
    out["substrate_depletion_percent"] = 100 - out.substrate_remaining_percent
    out["depletion_basis"] = np.where(
        depleted,
        "matched_control_signal_ratio",
        np.where(
            out.remaining_fraction_measurement_id.notna(),
            "reported_remaining_fraction",
            None,
        ),
    )
    # Negative net depletion is retained; values are never clipped to [0, 100].
    out["apparent_rate_s_inv"] = np.nan
    endpoint = depleted & out.time_s.gt(0)
    if endpoint.any():
        out.loc[endpoint, "apparent_rate_s_inv"] = (
            np.log(2)
            * out.loc[endpoint, "substrate_log2_depletion"]
            / out.loc[endpoint, "time_s"]
        )
    half = out.half_life.gt(0) & (~endpoint)
    out.loc[half, "apparent_rate_s_inv"] = np.log(2) / out.loc[half, "half_life"]
    out["apparent_rate_basis"] = np.where(
        endpoint,
        "exponential_control_relative_endpoint",
        np.where(half, "exponential_half_life", None),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        out["product_signal_increase_percent"] = 100 * np.expm1(
            np.log(2) * pd.to_numeric(out.product_log2_enrichment, errors="coerce")
        )
    out["relative_activity_percent"] = pd.to_numeric(
        out.relative_activity, errors="coerce"
    )
    out["relative_activity_basis"] = np.where(
        out.relative_activity.notna(), "reported_reference_normalized_activity", None
    )
    # Refit fluorescence in the intervals actually selected in the deposited sheet.
    source = dataset.table("source_records").set_index("record_id").raw_json
    qf = out.loc[out.source_db.eq("mmp_pics") & out.source_k_measurement_id.notna()]
    for column in [
        "fluorescence_rate_au_s",
        "fluorescence_fit_r_squared",
        "fluorescence_fit_start_s",
        "fluorescence_fit_end_s",
    ]:
        out[column] = np.nan
    if not qf.empty:
        signals = dataset.table("observations").loc[
            lambda x: x.quantity.eq("fluorescence_signal")
        ]
        trace_groups = {
            key: group
            for key, group in signals.groupby(
                ["assay_id", "subject_participant_id"], sort=False
            )
        }
        for index, row in qf.iterrows():
            raw = json.loads(source[row.source_k_source_record_id])
            start, end = raw.get("t1_s"), raw.get("t2_s")
            reported = raw.get("relative_activity_percent")
            if reported is not None and pd.notna(reported):
                out.loc[index, "relative_activity_percent"] = float(reported)
                out.loc[index, "relative_activity_basis"] = (
                    "source_round_rate_over_PLG_reference_times_100"
                )
            if start is None or end is None:
                continue
            trace = trace_groups.get((row.assay_id, row.subject_participant_id))
            if trace is None:
                continue
            values = (
                trace.loc[trace.time_s.between(start, end)]
                .groupby("time_s", sort=True)
                .value.mean()
            )
            if len(values) < 3:
                continue
            t = values.index.to_numpy(dtype=float)
            y = values.to_numpy(dtype=float)  # (T,), (T,)
            centered = t - t.mean()
            denominator = centered @ centered  # ()
            if denominator <= 0:
                continue
            slope = (centered @ (y - y.mean())) / denominator  # ()
            prediction = y.mean() + slope * centered  # (T,)
            residual = ((y - prediction) ** 2).sum()
            total = ((y - y.mean()) ** 2).sum()  # (), ()
            out.loc[
                index,
                [
                    "fluorescence_rate_au_s",
                    "fluorescence_fit_r_squared",
                    "fluorescence_fit_start_s",
                    "fluorescence_fit_end_s",
                ],
            ] = [slope, 1 - residual / total if total > 0 else np.nan, start, end]
        logger.info(
            "Computed %s source-interval fluorescence slopes",
            out.fluorescence_rate_au_s.notna().sum(),
        )
    numeric = [
        "kcat_km",
        "substrate_remaining_percent",
        "substrate_depletion_percent",
        "apparent_rate_s_inv",
        "product_signal_increase_percent",
    ]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return out
