"""Parse SABIO-RK reaction participants and normalized kinetic parameters."""

import json
import math
from pathlib import Path

from proteolysis_data_manager.integration.merge.contract.builder import ContractBuilder
from proteolysis_data_manager.integration.merge.parsers.kinetics._cleavage import (
    resolved_sites,
    species_peptide,
)
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

PARAMETERS = {
    "kcat": ("kcat", "s^(-1)"),
    "Km": ("km", "M"),
    "kcat/Km": ("kcat_km", "M^(-1)*s^(-1)"),
}


def _amount(parameters, name=None, role=None):
    """Read one explicitly matched, source-normalized concentration description.

    :param parameters: Raw kinetic-law parameter records.
    :param name: Compound name, or null for the unique catalyst concentration.
    :param role: Original reaction participant role.
    :return: Common participant amount fields; ambiguous matches remain unresolved.
    """
    matches = []
    for parameter in parameters:
        if (parameter.get("parameter_type") or {}).get("name") != "concentration":
            continue
        target = ((parameter.get("species") or {}).get("species_key") or "").split(
            " | "
        )
        if len(target) != 3 or target[2].lower() != role:
            continue
        if name is not None and target[1] != name:
            continue
        start = parameter.get("n_start_value")
        end = parameter.get("n_end_value")
        unit = (parameter.get("unit") or {}).get("n_name")
        if (
            unit != "M"
            or start is None
            or not math.isfinite(float(start))
            or float(start) < 0
        ):
            continue
        if end is None or end == start:
            matches.append({"amount": float(start), "unit": "M"})
        elif math.isfinite(float(end)) and float(end) >= float(start):
            matches.append({"lower": float(start), "upper": float(end), "unit": "M"})
    return matches[0] if len(matches) == 1 else {}


def _condition_value(info, temperature=False):
    value = info.get("start_value")
    if value is None or info.get("end_value") not in (None, value):
        return None
    if temperature:
        unit = info.get("unit")
        if unit in {"K", "Kelvin"}:
            return float(value) - 273.15
        if unit in {"°F", "Fahrenheit"}:
            return (float(value) - 32) * 5 / 9
        if unit not in {"°C", "Celsius"}:
            return None
    return float(value)


def parse(root, *, data=None):
    """Map source parameters without assigning one rate to multiple substrates.

    :param root: Acquired kinetics series root.
    :param data: Optional raw SABIO record iterable.
    :return: Common dataset with explicit participant and cut-attribution evidence.
    """
    logger.info("Mapping sabio source records to the common contract")
    root = Path(root)
    b = ContractBuilder(root.parent)
    seen = set()
    records = []
    if data is None:
        for path in sorted((root / "raw/sabio").glob("page_*.json")):
            records.extend((path, r) for r in json.loads(path.read_text())["data"])
    else:
        records = [(root / "raw/sabio", r) for r in data]
    for path, r in records:
        key = str(r["id"])
        if key in seen:
            raise ValueError("Duplicate SABIO source ID")
        seen.add(key)
        rid = b.source(
            "sabio",
            "reaction",
            key,
            path,
            f"entry/{key}",
            raw=r,
            publications=[f"PMID:{r['publication']['pubmed_id']}"]
            if (r.get("publication") or {}).get("pubmed_id")
            else [],
        )
        conditions = r.get("experimental_conditions") or {}
        assay = b.assay(
            rid,
            ("sabio", key),
            conditions={
                "ph": _condition_value(conditions.get("envvar_ph") or {}),
                "temperature_c": _condition_value(
                    conditions.get("envvar_temperature") or {}, True
                ),
                "buffer": conditions.get("buffer"),
                "mode": (r.get("general") or {}).get("experiment_type"),
            },
        )
        enzyme = r.get("enzyme_description") or {}
        general = r.get("general") or {}
        e = b.enzyme(
            rid,
            ("sabio", key),
            name=enzyme.get("enzyme_name"),
            ec=enzyme.get("ec_number"),
            uniprot=[
                p.get("uniprot_id")
                for p in enzyme.get("proteins") or []
                if p.get("uniprot_id")
            ],
            organism=(general.get("organism") or {}).get("name"),
            variant=enzyme.get("mutant_spec") or enzyme.get("wildtype"),
        )
        parameters = (r.get("kineticlaw") or {}).get("parameter") or []
        ep = b.participant(
            assay,
            e,
            "enzyme",
            enzyme=True,
            **_amount(parameters, role="catalyst"),
            status="reported_group_unresolved"
            if len(enzyme.get("proteins") or []) > 1
            else "reported",
        )
        species = (r.get("reaction") or {}).get("species") or []
        substrate_entries = []
        products = []
        for i, s in enumerate(species):
            role = (s.get("role") or "").lower()
            if role not in {"substrate", "product", "inhibitor", "activator"}:
                continue
            name = (s.get("compound") or {}).get("name")
            peptide = species_peptide(s)
            sub = b.substrate(
                rid,
                ("sabio", key, i),
                name=name,
                residues=peptide.residues if peptide else None,
                chemistry={"n_cap": peptide.n_cap, "c_cap": peptide.c_cap}
                if peptide
                else None,
            )
            participant = b.participant(
                assay, sub, role, **_amount(parameters, name, role)
            )
            if role == "substrate" and name not in {"H2O", "H+", "Water"}:
                substrate_entries.append((i, s, participant, peptide))
            if role == "product" and peptide:
                products.append(peptide)
        contexts = {}
        for i, s, participant, peptide in substrate_entries:
            if peptide is None:
                continue
            ids = []
            for cut, evidence in resolved_sites(
                peptide, s.get("comment") or "", products
            ):
                residues = [
                    peptide.residues[j] if 0 <= j < len(peptide.residues) else None
                    for j in range(cut - 4, cut + 4)
                ]
                ids.append(
                    b.context(
                        rid,
                        assay,
                        participant,
                        residues,
                        enzyme=ep,
                        cut=cut,
                        basis="parsed_substrate_residues",
                        evidence=evidence,
                        statuses=[
                            "observed" if t is not None else "outside_substrate"
                            for t in residues
                        ],
                    )
                )
            contexts[participant] = ids
        for pi, p in enumerate((r.get("kineticlaw") or {}).get("parameter") or []):
            kind = (p.get("parameter_type") or {}).get("name")
            if kind not in PARAMETERS:
                continue
            quantity, unit = PARAMETERS[kind]
            target = ((p.get("species") or {}).get("species_key") or "").split(" | ")
            matches = [
                entry
                for entry in substrate_entries
                if len(target) == 3
                and (entry[1].get("compound") or {}).get("name") == target[1]
                and target[2] == "Substrate"
            ]
            subject = (
                matches[0][2]
                if len(matches) == 1
                else (substrate_entries[0][2] if len(substrate_entries) == 1 else None)
            )
            value = p.get("n_start_value")
            end = p.get("n_end_value")
            valid = (p.get("unit") or {}).get("n_name") == unit
            point = (
                valid
                and value is not None
                and math.isfinite(float(value))
                and float(value) >= 0
            )
            interval = point and end not in (None, value)
            sd = p.get("n_standard_deviation") if valid else None
            mid = b.observation(
                rid,
                assay,
                subject,
                quantity,
                float(value) if point and not interval else None,
                key=pi,
                relation="interval" if interval else ("eq" if point else "unknown"),
                lower_bound=float(value) if interval else None,
                upper_bound=float(end) if interval else None,
                uncertainty_kind="sd" if sd is not None else None,
                uncertainty_value=sd,
                raw_value=str(p.get("start_value")),
                raw_unit=(p.get("unit") or {}).get("name"),
                derivation="source_normalized_parameter",
                measurement_scope="substrate_total" if subject else "unknown",
            )
            for cid in contexts.get(subject, []):
                b.link(mid, cid, "associated_context_not_site_specific_rate")
    return b.finish()
