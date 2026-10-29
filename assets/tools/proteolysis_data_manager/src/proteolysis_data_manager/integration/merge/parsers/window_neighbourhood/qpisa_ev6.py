"""Parse all EV6 source signals and their control-matched quantitative changes."""

from pathlib import Path

from proteolysis_data_manager.analysis.window_neighbourhood.qpisa import (
    POSITIONS,
    read_timecourse_measurements,
)
from proteolysis_data_manager.integration.merge.contract.builder import ContractBuilder
from proteolysis_data_manager.integration.merge.contract.schema import stable_id
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def parse(root, *, data=None):
    """Keep source substrate/product pair identities, times and control dependencies.

    :param root: Acquired window-neighbourhood series root.
    :param data: Optional dictionary from ``read_timecourse_measurements``.
    :return: Common dataset containing 168 signals and 72 relative changes.

    The primary methods pool HAE with DAP and DAE with HAP. Those shared reaction
    configurations are retained; pair identity never implies independent wells.
    """
    logger.info("Mapping qpisa_ev6 source records to the common contract")
    root = Path(root)
    q = (
        data
        if data is not None
        else read_timecourse_measurements(root / "raw/dpp4_qpisa")
    )
    b = ContractBuilder(root.parent)
    locators = {}
    setups = {}

    def setup(r):
        rid = b.source(
            "qpisa",
            "EV6",
            f"pair/{r['assay_pair']}/row/{r['source_excel_row']}",
            root / "raw/dpp4_qpisa" / r["source_file"],
            f"{r['source_sheet']}/row/{r['source_excel_row']}/pair/{r['assay_pair']}",
            raw=r,
            publications=[r["publication_doi"]],
        )
        pool = "HAE_DAP" if r["assay_pair"] in {1, 4} else "DAE_HAP"
        assay = b.assay(
            rid,
            ("qpisa", "EV6", pool, r["condition"], r["replicate"]),
            conditions={
                "temperature_c": 21.0,
                "buffer": "TBS; 2 mM TCEP",
                "method": "mass spectrometry",
                "mode": "in vitro",
            },
            basis="source_pooled_substrates_condition_replicate",
        )
        sid = b.substrate(
            rid,
            ("qpisa", "EV6", r["substrate_sequence"]),
            sequence=r["substrate_sequence"],
            residues=list(r["substrate_sequence"]),
        )
        sp = b.participant(assay, sid, "substrate", amount=1, unit="uM")
        pid = b.substrate(
            rid,
            ("qpisa", "EV6", r["product_sequence"]),
            sequence=r["product_sequence"],
            residues=list(r["product_sequence"]),
        )
        pp = b.participant(assay, pid, "product")
        ep = None
        if r["condition"] == "hDPP4":
            eid = b.enzyme(
                rid,
                ("qpisa", "EV6", "DPP4"),
                name="DPP4",
                code="S09.003",
                organism="Homo sapiens",
            )
            ep = b.participant(assay, eid, "enzyme", enzyme=True, amount=1, unit="ug")
        cid = b.context(
            rid,
            assay,
            sp,
            [r[p] for p in POSITIONS],
            enzyme=ep,
            cut=2,
            basis="substrate_residue_count",
            evidence="source_substrate_product_pair",
            statuses=["outside_substrate", "outside_substrate", *(["observed"] * 6)],
        )
        return rid, assay, sp, pp, cid

    for r in q["pairs"].to_dict("records"):
        rid, assay, sp, pp, cid = setup(r)
        locators[(r["assay_pair"], r["source_excel_row"])] = rid
        setups[(r["assay_pair"], r["source_excel_row"])] = (assay, sp, pp, cid)
        for role, subject, field, column in [
            ("substrate", sp, "substrate_signal_au", r["substrate_excel_column"]),
            ("product", pp, "product_signal_au", r["product_excel_column"]),
        ]:
            mid = b.observation(
                rid,
                assay,
                subject,
                "mass_spec_signal",
                r[field],
                key=role,
                time_s=r["time_s"],
                replicate_id=r["replicate"],
                raw_value=str(r[field]),
                raw_unit="instrument signal",
                derivation="reported_signal",
            )
            b.link(mid, cid, "source_defined_substrate_product_context")
    for r in q["changes"].to_dict("records"):
        rid = locators[(r["assay_pair"], r["source_excel_row"])]
        control = locators[(r["assay_pair"], r["control_source_excel_row"])]
        assay, sp, pp, cid = setups[(r["assay_pair"], r["source_excel_row"])]
        for quantity, subject, field in [
            ("substrate_log2_depletion", sp, "substrate"),
            ("product_log2_enrichment", pp, "product"),
        ]:
            mid = b.observation(
                rid,
                assay,
                subject,
                quantity,
                r[quantity],
                time_s=r["time_s"],
                replicate_id=r["replicate"],
                derivation="time_and_replicate_matched_log2_signal_ratio",
            )
            b.link(mid, cid, "source_substrate_product_pair")
            for role, srid, value in [
                ("treated_signal", rid, r[field + "_signal_au"]),
                ("control_signal", control, r["control_" + field + "_signal_au"]),
            ]:
                b.add(
                    "observation_inputs",
                    input_id=stable_id("input", mid, role),
                    measurement_id=mid,
                    role=role,
                    value=value,
                    unit="instrument_au",
                    source_record_id=srid,
                    source_field=field + "_signal_au",
                )
    return b.finish()
