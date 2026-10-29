"""Build and validate a common dataset while keeping source experiment identities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from proteolysis_data_manager.integration.merge.contract.schema import (
    PRIMARY_KEYS,
    QUANTITIES,
    TABLES,
    canonical,
    plain,
    stable_id,
)
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


from proteolysis_data_manager.integration.merge.contract.dataset import UnifiedDataset


class ContractBuilder:
    """Accumulate source records, shared condition descriptions and observations.

    :param data_root: Parent directory of the acquired data-series directories.
    :type data_root: str | pathlib.Path | None
    """

    def __init__(self, data_root=None):
        self.data_root = Path(data_root).resolve() if data_root else None
        self.rows = {name: {} for name in TABLES}

    def add(self, table, **record):
        """Insert an idempotent row; conflicting reuse of an ID is an error.

        :param str table: Contract table name.
        :param record: Named contract column values.
        :return: Primary key scalar or tuple.
        :raises ValueError: If columns or the same primary key conflict.
        """
        if unknown := set(record) - set(TABLES[table]):
            raise ValueError(f"Unknown {table} columns: {sorted(unknown)}")
        row = {column: plain(record.get(column)) for column in TABLES[table]}
        key = tuple(row[column] for column in PRIMARY_KEYS[table])
        previous = self.rows[table].get(key)
        if previous is not None and previous != row:
            # Shared descriptions retain their first provenance anchor; all observation
            # source records remain separately linked. Scientific fields must agree.
            ignored = (
                {"source_record_id"}
                if table in {"assays", "substrates", "enzymes", "contexts"}
                else set()
            )
            if any(previous[c] != row[c] for c in row if c not in ignored):
                raise ValueError(f"Conflicting {table} primary key {key}")
            return key[0] if len(key) == 1 else key
        self.rows[table][key] = row
        return key[0] if len(key) == 1 else key

    def source(
        self, db, dataset, source_id, path, locator, *, raw=None, publications=None
    ):
        """Register original identity and locator, retaining source values.

        :param str db: Database/publication source name.
        :param str dataset: Original source subset or workbook role.
        :param source_id: Original record identity within that dataset.
        :param path: Original raw file path.
        :param str locator: JSON pointer, worksheet row, or archive member locator.
        :param raw: Original source fields used by the parser.
        :param publications: Reported bibliographic identifiers.
        :return: Source record ID.
        """
        path = Path(path).resolve()
        if (
            path.is_absolute()
            and self.data_root
            and path.is_relative_to(self.data_root)
        ):
            path = path.relative_to(self.data_root)
        record_id = stable_id("source", db, dataset, str(source_id), str(path), locator)
        return self.add(
            "source_records",
            record_id=record_id,
            source_db=db,
            source_dataset=dataset,
            source_id=str(source_id),
            raw_path=path.as_posix(),
            locator=locator,
            publication_ids_json=canonical(publications or []),
            raw_json=canonical(raw or {}),
        )

    def conditions(self, **fields):
        """Reuse identical recorded condition descriptions, independently of assays.

        :param fields: Structured condition fields; unknown values remain null.
        :return: Shared condition-description ID, not an experiment identity.
        """
        row = {
            column: fields.get(column)
            for column in TABLES["condition_sets"]
            if column != "condition_id"
        }
        for column, value in row.items():
            if isinstance(value, str) and value.lower().strip() in {
                "unknown",
                "other",
                "not reported",
                "",
            }:
                row[column] = None
        return self.add(
            "condition_sets", condition_id=stable_id("conditions", row), **row
        )

    def assay(self, record_id, key, *, conditions=None, basis="source_record"):
        """Create an explicitly scoped assay, never merge by equal conditions alone.

        :param str record_id: Source identity supporting the grouping.
        :param key: Source-defined assay/setup identity.
        :param conditions: Structured shared condition description.
        :param str basis: Evidence for the assay grouping.
        :return: Assay ID.
        """
        return self.add(
            "assays",
            assay_id=stable_id("assay", key),
            condition_id=self.conditions(**(conditions or {})),
            source_record_id=record_id,
            grouping_basis=basis,
        )

    def substrate(
        self,
        record_id,
        key,
        *,
        name=None,
        sequence=None,
        residues=None,
        chemistry=None,
        smiles=None,
    ):
        """Register a source-scoped molecular description without guessed equivalence.

        :param str record_id: Source record supporting the description.
        :param key: Source-scoped substrate identity.
        :param name: Original substrate name.
        :param sequence: Source sequence notation, preserving stereochemical case.
        :param residues: Resolved residue tokens, including nonstandard chemistry.
        :param chemistry: Original chirality, topology and modification fields.
        :param smiles: Reported structural representation, without reconstruction.
        :return: Substrate ID.
        """
        description = {
            "sequence": sequence,
            "residues": residues,
            "chemistry": chemistry or {},
            "smiles": smiles,
        }
        return self.add(
            "substrates",
            substrate_id=stable_id("substrate", key),
            name=name,
            sequence=sequence,
            residues_json=canonical(residues) if residues is not None else None,
            chemistry_json=canonical(chemistry or {}),
            smiles=smiles,
            description_key=stable_id("description", description),
            source_record_id=record_id,
        )

    def enzyme(
        self,
        record_id,
        key,
        *,
        name=None,
        code=None,
        ec=None,
        uniprot=None,
        organism=None,
        variant=None,
    ):
        """Register the supplied enzyme identity without EC-to-MEROPS guessing.

        :param str record_id: Supporting source record.
        :param key: Source enzyme identity, including reported variants.
        :param name: Reported enzyme name.
        :param code: Reported/reviewed MEROPS mapping.
        :param ec: Reported EC number.
        :param uniprot: Reported UniProt accessions.
        :param organism: Enzyme organism, not the assay-medium species.
        :param variant: Reported variant description.
        :return: Enzyme ID.
        """
        return self.add(
            "enzymes",
            enzyme_id=stable_id("enzyme", key),
            name=name,
            merops_code=code,
            ec_number=ec,
            uniprot_json=canonical(uniprot or []),
            organism=organism,
            variant=variant,
            source_record_id=record_id,
        )

    def participant(
        self,
        assay_id,
        entity_id,
        role,
        *,
        enzyme=False,
        amount=None,
        lower=None,
        upper=None,
        unit=None,
        status="reported",
    ):
        """Associate an entity and amount with one concrete assay configuration.

        :param str assay_id: Source-scoped assay identity.
        :param str entity_id: Enzyme or substrate identity.
        :param str role: Substrate, product, enzyme, inhibitor, or reference role.
        :param bool enzyme: Whether the entity is an enzyme.
        :param amount: Reported amount or concentration point.
        :param lower: Reported amount lower bound.
        :param upper: Reported amount upper bound.
        :param unit: Amount unit.
        :param str status: Participant-attribution status.
        :return: Participant ID.
        """
        reported = canonical(
            {"amount": amount, "lower": lower, "upper": upper, "unit": unit}
        )
        canonical_unit, factor = {
            "M": ("M", 1),
            "mM": ("M", 1e-3),
            "uM": ("M", 1e-6),
            "µM": ("M", 1e-6),
            "nM": ("M", 1e-9),
            "pM": ("M", 1e-12),
            "kg": ("kg", 1),
            "g": ("kg", 1e-3),
            "mg": ("kg", 1e-6),
            "ug": ("kg", 1e-9),
            "µg": ("kg", 1e-9),
            "ng": ("kg", 1e-12),
        }.get(unit, (unit, 1))
        return self.add(
            "participants",
            participant_id=stable_id("participant", assay_id, entity_id, role),
            assay_id=assay_id,
            enzyme_id=entity_id if enzyme else None,
            substrate_id=None if enzyme else entity_id,
            role=role,
            amount=amount * factor if amount is not None else None,
            amount_lower=lower * factor if lower is not None else None,
            amount_upper=upper * factor if upper is not None else None,
            amount_unit=canonical_unit,
            reported_amount_json=reported,
            link_status=status,
        )

    def observation(
        self, record_id, assay_id, subject, quantity, value, *, key=None, **fields
    ):
        """Store a quantity in its registered unit, retaining unresolved source values.

        :param str record_id: Supporting source record.
        :param str assay_id: Source assay identity.
        :param subject: Measured substrate/product participant, or null if unresolved.
        :param str quantity: Registered common quantity name.
        :param value: Reported/derived point, or null for bounds/unresolved values.
        :param key: Stable source measurement key.
        :param fields: Bounds, uncertainty, timing, scope and provenance details.
        :return: Measurement ID.
        """
        return self.add(
            "observations",
            measurement_id=stable_id("measurement", record_id, quantity, key),
            assay_id=assay_id,
            subject_participant_id=subject,
            quantity=quantity,
            value=value,
            unit=QUANTITIES[quantity][0],
            relation=fields.pop("relation", "eq" if value is not None else "unknown"),
            measurement_scope=fields.pop("measurement_scope", "substrate_total"),
            source_record_id=record_id,
            **fields,
        )

    def context(
        self,
        record_id,
        assay_id,
        subject,
        residues,
        *,
        enzyme=None,
        cut=None,
        basis=None,
        evidence,
        statuses=None,
    ):
        """Store exactly eight tokens with distinct unavailable/unresolved statuses.

        :param str record_id: Original source context locator.
        :param str assay_id: Assay identity.
        :param subject: Known parent-substrate participant, or null for window-only evidence.
        :param residues: Eight P4–P4′ tokens, preserving nonstandard residues.
        :param enzyme: Enzyme participant, if unambiguously attributed.
        :param cut: Number of parent residues before the cut, if known.
        :param basis: Coordinate convention of ``cut``.
        :param str evidence: Source annotation or explicit reconstruction method.
        :param statuses: Eight position statuses; null tokens default to unresolved.
        :return: Context ID.
        """
        residues = list(residues)
        statuses = statuses or [
            "observed" if r is not None else "unresolved" for r in residues
        ]
        return self.add(
            "contexts",
            context_id=stable_id(
                "context", assay_id, subject, enzyme, cut, residues, evidence
            ),
            assay_id=assay_id,
            substrate_participant_id=subject,
            enzyme_participant_id=enzyme,
            cut_after_residue=cut,
            coordinate_basis=basis,
            residues_json=canonical(residues),
            position_status_json=canonical(statuses),
            site_evidence=evidence,
            source_record_id=record_id,
        )

    def link(self, measurement_id, context_id, attribution="associated_context"):
        """Link without copying the numeric observation onto each possible site.

        :param str measurement_id: Numeric observation identity.
        :param str context_id: Associated cut-context identity.
        :param str attribution: Site-specific or context-association evidence.
        :return: Link key.
        """
        return self.add(
            "observation_contexts",
            measurement_id=measurement_id,
            context_id=context_id,
            attribution=attribution,
        )

    def finish(self):
        """Create typed public tables and validate their relationships.

        :return: A validated unified dataset.
        """
        return UnifiedDataset(
            {
                name: pd.DataFrame(list(rows.values()), columns=TABLES[name])
                for name, rows in self.rows.items()
            }
        ).validate()
