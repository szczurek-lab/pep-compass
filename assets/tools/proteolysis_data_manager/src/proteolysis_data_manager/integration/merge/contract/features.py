"""Names and units of the public wide feature columns."""

from proteolysis_data_manager.integration.merge.contract.schema import QUANTITIES

FEATURE_UNITS = {name: unit for name, (unit, _, _) in QUANTITIES.items()}
FEATURE_UNITS.update(
    {
        "substrate_remaining_percent": "percent_control",
        "substrate_depletion_percent": "percent_control",
        "product_signal_increase_percent": "percent_control",
        "relative_activity_percent": "percent_PLG_reference",
        "apparent_rate_s_inv": "s^-1",
        "fluorescence_rate_au_s": "instrument_au s^-1",
        "fluorescence_fit_r_squared": "dimensionless",
        "position_probability_sum": "probability_sum",
        **{
            "p_" + position: "probability"
            for position in ("P4", "P3", "P2", "P1", "P1p", "P2p", "P3p", "P4p")
        },
    }
)
