"""Common qPISA endpoint mapping for independently selectable source experiments."""

from functools import lru_cache
from pathlib import Path

from proteolysis_data_manager.analysis.window_neighbourhood.qpisa import (
    read_endpoint_measurements,
)
from proteolysis_data_manager.integration.merge.contract.builder import ContractBuilder
from proteolysis_data_manager.integration.merge.contract.schema import (
    POSITIONS,
    stable_id,
)
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@lru_cache(maxsize=4)
def _load(root, manifest):
    return read_endpoint_measurements(Path(root) / "raw/dpp4_qpisa")


def parse_experiment(root, experiment, *, data=None):
    """Normalize every signed source response and preserve its two input levels.

    :param root: Acquired window-neighbourhood series root.
    :param str experiment: EV1 or EV4.
    :param data: Optional already-loaded endpoint frame for that experiment or both.
    :return: Common dataset with source inputs, partial windows and parent evidence.
    """
    logger.info("Mapping qpisa_endpoint source records to the common contract")
    root = Path(root)
    frame = (
        data
        if data is not None
        else _load(str(root.resolve()), (root / "raw/manifest.json").read_bytes())
    )
    frame = frame.loc[frame.experiment.eq(experiment)]
    b = ContractBuilder(root.parent)
    for r in frame.to_dict("records"):
        rid = b.source(
            "qpisa",
            experiment,
            r["source_record_id"],
            root / "raw/dpp4_qpisa" / r["source_file"],
            f"{r['source_sheet']}/row/{r['source_excel_row']}",
            raw=r,
            publications=[r["publication_doi"]],
        )
        assay = b.assay(
            rid,
            ("qpisa", experiment, r["enzyme"]),
            conditions={
                "ph": r["ph"],
                "temperature_c": r["temperature_c"],
                "buffer": "20 mM HEPES; 150 mM NaCl; 2 mM TCEP",
                "method": "qPISA TMT mass spectrometry",
                "mode": "in vitro",
            },
            basis="source_mixture_experiment_enzyme",
        )
        eid = b.enzyme(
            rid,
            ("qpisa", experiment, r["enzyme"]),
            name=r["enzyme"],
            code=r["merops_code"],
            organism="Homo sapiens"
            if r["enzyme"] == "DPP4"
            else "Caenorhabditis elegans",
        )
        ep = b.participant(
            assay, eid, "enzyme", enzyme=True, amount=r["enzyme_mass_ug"], unit="ug"
        )
        sid = b.substrate(
            rid,
            ("qpisa", experiment, r["measured_sequence"]),
            sequence=r["measured_sequence"],
            residues=list(r["measured_sequence"]),
        )
        sp = b.participant(
            assay, sid, "substrate" if r["status"] == "Tryptic+" else "product"
        )
        mid = b.observation(
            rid,
            assay,
            sp,
            r["measurement_type"],
            r["value"],
            time_s=r["duration_s"],
            raw_value=str(r["value"]),
            raw_unit=r["unit"],
            derivation="difference_of_reported_log2_levels",
        )
        for role, field in [
            ("treated_level", "enzyme_log2_level"),
            ("control_level", "control_log2_level"),
        ]:
            b.add(
                "observation_inputs",
                input_id=stable_id("input", mid, role),
                measurement_id=mid,
                role=role,
                value=r[field],
                unit="normalized_log2_signal",
                source_record_id=rid,
                source_field=r["enzyme_source_column"]
                if role == "treated_level"
                else r["control_source_column"],
            )
        if r["has_window"]:
            if r["status"] == "Tryptic+":
                parent = sp
            else:
                parent_id = b.substrate(
                    rid,
                    ("qpisa", experiment, r["substrate_sequence"]),
                    sequence=r["substrate_sequence"],
                    residues=list(r["substrate_sequence"]),
                )
                parent = b.participant(assay, parent_id, "substrate")
            cid = b.context(
                rid,
                assay,
                parent,
                [r[p] for p in POSITIONS],
                enzyme=ep,
                cut=2,
                basis="substrate_residue_count",
                evidence=r["site_evidence"],
                statuses=[
                    "outside_substrate",
                    "outside_substrate",
                    *(["observed"] * 6),
                ],
            )
            b.link(
                mid, cid, "geometry_or_sequence_association_not_independent_cut_count"
            )
    return b.finish()
