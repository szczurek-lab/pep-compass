"""Parse all acquired EC 3.4 kinetic entries, retaining source-supported contexts."""

import re
from pathlib import Path

from proteolysis_data_manager.integration.kinetics.adapters import brenda as original
from proteolysis_data_manager.integration.merge.contract.builder import ContractBuilder
from proteolysis_data_manager.integration.merge.parsers.kinetics._cleavage import (
    parse_peptide,
    resolved_sites,
)
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def parse(root, *, data=None):
    """Map BRENDA kinetics to registered SI quantities and source scopes.

    :param root: Acquired kinetics series root.
    :param data: Optional iterable of ``(EC, raw block)`` records.
    :return: Common dataset; each raw measurement is kept exactly once.
    """
    root = Path(root)
    archive = next((root / "raw/brenda").glob("*.tar.gz"))
    b = ContractBuilder(root.parent)
    invalid_ranges = 0
    logger.info("Reading acquired BRENDA kinetic entries")
    records = data if data is not None else original.iter_records(archive)
    for ec, block in records:
        if not ec.startswith("3.4."):
            continue
        protein_block = block.get("protein") or {}
        references = block.get("reference") or {}
        for family, (quantity, raw_unit, factor) in original.PARAMETERS.items():
            reported_unit = {
                "turnover_number": "s^-1",
                "km_value": "mM",
                "kcat_km_value": "mM^-1 s^-1",
            }[family]
            for index, entry in enumerate(block.get(family) or []):
                locator = f"data/{ec}/{family}/{index}"
                pubs = [
                    references.get(str(key), {})
                    for key in entry.get("references") or []
                ]
                rid = b.source(
                    "brenda",
                    family,
                    locator,
                    archive,
                    locator,
                    raw={"entry": entry, "references": pubs, "ec": ec},
                    publications=[f"PMID:{p['pmid']}" for p in pubs if p.get("pmid")],
                )
                comment = str(entry.get("comment") or "")
                assay = b.assay(
                    rid,
                    ("brenda", locator),
                    conditions={
                        "ph": original.scalar_condition(comment, "ph"),
                        "temperature_c": original.scalar_condition(
                            comment, "temperature_c"
                        ),
                    },
                )
                enzyme_participants = []
                for key in entry.get("proteins") or []:
                    protein = protein_block.get(str(key)) or {}
                    variant = original.reported_variant(comment)
                    e = b.enzyme(
                        rid,
                        ("brenda", ec, str(key), variant),
                        name=original.text(block.get("recommended_name")),
                        ec=ec,
                        uniprot=protein.get("accessions"),
                        organism=original.text(protein.get("organism")),
                        variant=variant,
                    )
                    enzyme_participants.append(
                        b.participant(
                            assay,
                            e,
                            "enzyme",
                            enzyme=True,
                            status="direct"
                            if len(entry["proteins"]) == 1
                            else "reported_group_unresolved",
                        )
                    )
                text, label = original.split_value(str(entry.get("value") or ""))
                peptide = parse_peptide(label) if label else None
                participant = None
                if label:
                    substrate = b.substrate(
                        rid,
                        ("brenda", ec, label),
                        name=label,
                        sequence="".join(peptide.residues)
                        if peptide and all(len(t) == 1 for t in peptide.residues)
                        else None,
                        residues=peptide.residues if peptide else None,
                        chemistry={"n_cap": peptide.n_cap, "c_cap": peptide.c_cap}
                        if peptide
                        else None,
                    )
                    participant = b.participant(assay, substrate, "substrate")
                value, relation = original.parse_value(text)
                error = re.fullmatch(
                    rf"\s*({original.NUMBER})\s*(?:±|\+/-)\s*({original.NUMBER})\s*",
                    text,
                )
                uncertainty = (
                    float(error[2]) * factor if error and float(error[2]) >= 0 else None
                )
                lower = upper = None
                if relation in {"lt", "le", "gt", "ge", "range"}:
                    numbers = re.findall(original.NUMBER, text)
                    if relation == "range":
                        # Parse the separator independently of either signed endpoint.
                        match = re.fullmatch(
                            rf"\s*({original.NUMBER})\s*(?:-|–|to)\s*({original.NUMBER})\s*",
                            text,
                        )
                        if match:
                            lower, upper = [float(n) * factor for n in match.groups()]
                            relation = "interval"
                    elif len(numbers) == 1:
                        if relation in {"lt", "le"}:
                            upper = float(numbers[0]) * factor
                        else:
                            lower = float(numbers[0]) * factor
                if relation == "approx":
                    numbers = re.findall(original.NUMBER, text)
                    value = float(numbers[0]) if len(numbers) == 1 else None
                derivation = "source_unit_conversion" if factor != 1 else "reported"
                if lower is not None and upper is not None and lower > upper:
                    # REMARK: The source contains descending ranges ending in zero.
                    # Their intended limits are unknown; retain the literal raw value.
                    lower = upper = value = None
                    relation = "unknown"
                    invalid_ranges += 1
                    derivation = "unresolved_source_descending_range"
                mid = b.observation(
                    rid,
                    assay,
                    participant,
                    quantity,
                    value * factor if value is not None else None,
                    relation=relation,
                    lower_bound=lower,
                    upper_bound=upper,
                    measurement_scope="unknown",
                    uncertainty_kind="reported_unspecified"
                    if uncertainty is not None
                    else None,
                    uncertainty_value=uncertainty,
                    raw_value=text,
                    raw_unit=reported_unit,
                    derivation=derivation,
                )
                if peptide:
                    for cut, evidence in resolved_sites(peptide, comment):
                        indices = range(cut - 4, cut + 4)
                        residues = [
                            peptide.residues[i]
                            if 0 <= i < len(peptide.residues)
                            else None
                            for i in indices
                        ]
                        statuses = [
                            "observed" if r is not None else "outside_substrate"
                            for r in residues
                        ]
                        context = b.context(
                            rid,
                            assay,
                            participant,
                            residues,
                            enzyme=enzyme_participants[0]
                            if len(enzyme_participants) == 1
                            else None,
                            cut=cut,
                            basis="parsed_substrate_residues",
                            evidence=evidence,
                            statuses=statuses,
                        )
                        b.link(
                            mid, context, "associated_context_not_site_specific_rate"
                        )
    if invalid_ranges:
        logger.warning(
            "Retained %s descending BRENDA ranges as unresolved source values",
            invalid_ranges,
        )
    return b.finish()
