"""Reuse reviewed source adapters, including source values absent from flat views."""

from functools import lru_cache
from pathlib import Path

from proteolysis_data_manager.integration.half_life.adapters.common import (
    condition_frame,
)
from proteolysis_data_manager.integration.half_life.datasets import load_adapter_records
from proteolysis_data_manager.integration.merge.contract.builder import ContractBuilder
from proteolysis_data_manager.integration.merge.contract.schema import plain
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@lru_cache(maxsize=4)
def _records(root, manifest_bytes):
    # Cache within this immutable acquisition snapshot; the existing loader verifies hashes.
    return condition_frame(
        __import__("pandas").DataFrame(load_adapter_records(Path(root)))
    ).to_dict("records")


def parse_source(root, source, *, data=None):
    """Map one reviewed half-life adapter into the common observation contract.

    :param root: Half-time series root.
    :param str source: Source adapter identifier.
    :param data: Optional already-loaded complete adapter inventory.
    :return: Common dataset, retaining bounds, approximate values and other readouts.
    """
    logger.info("Mapping half_life source records to the common contract")
    root = Path(root)
    rows = (
        data
        if data is not None
        else _records(str(root.resolve()), (root / "raw/manifest.json").read_bytes())
    )
    b = ContractBuilder(root.parent)
    for r in rows:
        if r["source"] != source:
            continue
        r = plain(r)
        rid = b.source(
            source,
            r.get("dataset_view", "source"),
            r["observation_id"],
            root / "raw" / source / r["source_filename"],
            r.get("source_locator") or r["observation_id"],
            raw=r,
            publications=[r.get("publication_id")] if r.get("publication_id") else [],
        )
        conditions = {
            "sample_species": r.get("species"),
            "matrix": r.get("matrix"),
            "matrix_fraction": r.get("matrix_fraction"),
            "ph": r.get("ph"),
            "temperature_c": r.get("temperature_c"),
            "method": r.get("assay"),
            "mode": r.get("in_vivo_in_vitro"),
        }
        assay = b.assay(
            rid, ("half_life", source, r["observation_id"]), conditions=conditions
        )
        chemistry = {
            key: r.get(key)
            for key in [
                "chirality",
                "topology",
                "n_terminal_modification",
                "c_terminal_modification",
                "chemical_modifications",
            ]
        }
        substrate = b.substrate(
            rid,
            (source, r["observation_id"]),
            name=r.get("name"),
            sequence=r.get("sequence_raw"),
            chemistry=chemistry,
            smiles=r.get("smiles"),
        )
        participant = b.participant(
            assay,
            substrate,
            "substrate",
            amount=r.get("peptide_concentration_um"),
            unit="uM" if r.get("peptide_concentration_um") is not None else None,
        )
        quantity = r.get("measurement_kind", "half_life")
        if quantity not in {"half_life", "remaining_fraction", "functional_activity"}:
            quantity = "functional_activity"
        relation = r.get("relation", "unknown")
        lower = r.get("lower_bound_seconds") if quantity == "half_life" else None
        upper = r.get("upper_bound_seconds") if quantity == "half_life" else None
        value = r.get("half_life_seconds")
        if relation == "approx":
            value = r.get("approximate_half_life_seconds")
        if quantity == "remaining_fraction":
            value = r.get("remaining_fraction")
            relation = r.get("fraction_relation", "unknown")
        elif quantity != "half_life":
            value = None
            relation = "unknown"
        if quantity == "remaining_fraction" and relation not in {"eq", "approx"}:
            if relation in {"gt", "ge"}:
                lower = value
            if relation in {"lt", "le"}:
                upper = value
            value = None
        b.observation(
            rid,
            assay,
            participant,
            quantity,
            value,
            relation=relation,
            lower_bound=lower,
            upper_bound=upper,
            uncertainty_kind=r.get("uncertainty_kind"),
            uncertainty_value=r.get("reported_error_seconds"),
            ci_lower=r.get("ci_lower_seconds"),
            ci_upper=r.get("ci_upper_seconds"),
            ci_level=r.get("ci_confidence_level"),
            replicate_count=r.get("n_replicates"),
            time_s=r.get("observation_time_seconds"),
            equivalence_group=r.get("representation_key"),
            raw_value=r.get("half_life_raw"),
            raw_unit=r.get("unit_raw"),
            derivation="reported",
        )
    return b.finish()
