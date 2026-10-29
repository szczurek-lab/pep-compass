"""Parse every reported QF coefficient, activity and raw time-series signal."""

from pathlib import Path

from proteolysis_data_manager.analysis.window_neighbourhood.pics import (
    MMP_TO_MEROPS,
    POSITIONS,
    qf_measurements_with_windows,
    read_qf_assay,
)
from proteolysis_data_manager.integration.merge.contract.builder import ContractBuilder
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def parse(root, *, data=None):
    """Map source quantities without refitting the experiment.

    :param root: Acquired window-neighbourhood series root.
    :param data: Optional dictionary returned by ``read_qf_assay``.
    :return: Common dataset containing all coefficients, summaries and signals.
    """
    logger.info("Mapping mmp_pics_qf source records to the common contract")
    root = Path(root)
    path = root / "raw/mmp_pics/mmc8.xlsx"
    qf = data if data is not None else read_qf_assay(path)
    measured = qf_measurements_with_windows(qf)
    b = ContractBuilder(root.parent)
    chemistry = qf["substrates"].set_index("substrate").to_dict("index")

    def setup(rid, enzyme, label, round_id):
        spec = chemistry[label]
        assay = b.assay(
            rid,
            ("PICS_QF", enzyme, label, str(round_id)),
            conditions={
                "ph": 7.5,
                "temperature_c": 37.0,
                "buffer": "100 mM Tris; 100 mM NaCl; 10 mM CaCl2; 0.05% Brij-35",
                "method": "quenched fluorescence",
                "mode": "in vitro",
            },
            basis="source_enzyme_substrate_analysis_round"
            if round_id != "summary"
            else "source_merged_summary",
        )
        sid = b.substrate(
            rid,
            ("PICS_QF", label),
            name=spec["full_source_chemistry"],
            residues=[spec[p] for p in POSITIONS if spec[p] is not None],
            chemistry={
                "n_cap": "Mca",
                "c_cap": "NH2",
                "chirality": "L",
                "topology": "linear",
                "P2p_residue": "Dpa",
            },
        )
        sp = b.participant(assay, sid, "substrate", amount=1, unit="uM")
        eid = b.enzyme(
            rid,
            ("PICS_QF", enzyme),
            name=enzyme,
            code=MMP_TO_MEROPS[enzyme],
            organism="Homo sapiens",
        )
        ep = b.participant(
            assay, eid, "enzyme", enzyme=True, lower=1, upper=10, unit="nM"
        )
        cid = b.context(
            rid,
            assay,
            sp,
            [spec[p] for p in POSITIONS],
            enzyme=ep,
            cut=3,
            basis="substrate_residue_count",
            evidence="source_cut_annotation",
            statuses=["outside_substrate", *(["observed"] * 7)],
        )
        return assay, sp, cid

    for r in measured.to_dict("records"):
        rid = b.source(
            "mmp_pics",
            "QF",
            r["source_record_id"],
            path,
            f"{r['sheet']}/{r['source_value_cell']}",
            raw=r,
            publications=[r["publication_doi"]],
        )
        round_id = (
            r.get("analysis_round")
            if r["measurement_type"] == "source_k"
            else "summary"
        )
        assay, sp, cid = setup(rid, r["enzyme"], r["substrate"], round_id)
        mid = b.observation(
            rid,
            assay,
            sp,
            r["measurement_type"],
            r["value"],
            uncertainty_kind="sd"
            if r["measurement_type"] == "relative_activity"
            else None,
            uncertainty_value=r.get("value_sd")
            if r["measurement_type"] == "relative_activity"
            else None,
            raw_value=str(r["value"]),
            raw_unit=r.get("unit"),
            measurement_scope="site_specific",
            derivation="reported_coefficient_or_summary",
        )
        b.link(mid, cid, "source_defined_cut")
    for r in qf["traces"].to_dict("records"):
        rid = b.source(
            "mmp_pics",
            "QF_trace",
            f"{r['sheet']}:{r['excel_row']}:{r['excel_column']}",
            path,
            f"{r['sheet']}/row/{r['excel_row']}/column/{r['excel_column']}",
            raw=r,
            publications=["DOI:10.1016/j.matbio.2015.09.003"],
        )
        assay, sp, cid = setup(
            rid, r["enzyme"], r["substrate"], float(r["analysis_round"])
        )
        mid = b.observation(
            rid,
            assay,
            sp,
            "fluorescence_signal",
            r["fluorescence_au"],
            time_s=r["time_s"],
            replicate_id=r["well"],
            raw_value=str(r["fluorescence_au"]),
            raw_unit="a.u.",
            derivation="reported_signal",
        )
        b.link(mid, cid, "source_defined_substrate_context")
    return b.finish()
