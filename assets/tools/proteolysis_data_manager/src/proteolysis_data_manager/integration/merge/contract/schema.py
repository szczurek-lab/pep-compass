"""Versioned common observation contract and source-independent quantity definitions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

SCHEMA_VERSION = "1.0"
POSITIONS = ("P4", "P3", "P2", "P1", "P1p", "P2p", "P3p", "P4p")
AA = "ACDEFGHIKLMNPQRSTVWY"
# Unit and sign semantics belong to the quantity, not to a source adapter.
QUANTITIES = {
    "half_life": (
        "s",
        False,
        "Reported peptide half-life; limits and approximations remain explicit.",
    ),
    "remaining_fraction": (
        "fraction",
        False,
        "Remaining substrate fraction at an explicitly recorded time.",
    ),
    "functional_activity": (
        None,
        True,
        "Reported functional activity; unresolved units remain null.",
    ),
    "kcat": ("s^-1", False, "Catalytic turnover number."),
    "km": ("M", False, "Michaelis constant."),
    "kcat_km": ("M^-1 s^-1", False, "Catalytic efficiency."),
    "source_k": (
        None,
        True,
        "Coefficient named k by the source; no calibrated unit is invented.",
    ),
    "relative_activity": (
        "percent_reference",
        True,
        "Activity relative to the recorded reference substrate.",
    ),
    "substrate_log2_depletion": (
        "log2_ratio",
        True,
        "Control minus enzyme log2 signal.",
    ),
    "product_log2_enrichment": (
        "log2_ratio",
        True,
        "Enzyme minus control log2 signal.",
    ),
    "fluorescence_signal": (
        "instrument_au",
        True,
        "Uncalibrated source fluorescence signal.",
    ),
    "mass_spec_signal": (
        "instrument_au",
        False,
        "Uncalibrated source mass-spectrometry signal.",
    ),
    "cleavage_occurrence": (
        "count",
        False,
        "One source-reported cleavage occurrence row, not a kinetic rate.",
    ),
    "merops_count": ("count", False, "Original marginal residue occurrence count."),
    "merops_log_probability": (
        "ln_probability",
        True,
        "Original source log-probability of a residue at a position.",
    ),
    "merops_probability_sum": (
        "probability_sum",
        False,
        "Sum of exp(source log-probability) over available positions.",
    ),
    "merops_log_probability_sum": (
        "ln_probability",
        True,
        "Sum of source log-probabilities over available positions.",
    ),
    "merops_probability_product": (
        "probability",
        False,
        "Product of positional probabilities under an independent-position representation.",
    ),
}
# SQL types also define the public table column order.
TABLES = {
    "source_records": {
        "record_id": "TEXT",
        "source_db": "TEXT",
        "source_dataset": "TEXT",
        "source_id": "TEXT",
        "raw_path": "TEXT",
        "locator": "TEXT",
        "publication_ids_json": "TEXT",
        "raw_json": "TEXT",
    },
    "condition_sets": {
        "condition_id": "TEXT",
        "sample_species": "TEXT",
        "matrix": "TEXT",
        "matrix_fraction": "REAL",
        "ph": "REAL",
        "temperature_c": "REAL",
        "buffer": "TEXT",
        "method": "TEXT",
        "mode": "TEXT",
    },
    "assays": {
        "assay_id": "TEXT",
        "condition_id": "TEXT",
        "source_record_id": "TEXT",
        "grouping_basis": "TEXT",
    },
    "substrates": {
        "substrate_id": "TEXT",
        "name": "TEXT",
        "sequence": "TEXT",
        "residues_json": "TEXT",
        "chemistry_json": "TEXT",
        "smiles": "TEXT",
        "description_key": "TEXT",
        "source_record_id": "TEXT",
    },
    "enzymes": {
        "enzyme_id": "TEXT",
        "name": "TEXT",
        "merops_code": "TEXT",
        "ec_number": "TEXT",
        "uniprot_json": "TEXT",
        "organism": "TEXT",
        "variant": "TEXT",
        "source_record_id": "TEXT",
    },
    "participants": {
        "participant_id": "TEXT",
        "assay_id": "TEXT",
        "substrate_id": "TEXT",
        "enzyme_id": "TEXT",
        "role": "TEXT",
        "amount": "REAL",
        "amount_lower": "REAL",
        "amount_upper": "REAL",
        "amount_unit": "TEXT",
        "reported_amount_json": "TEXT",
        "link_status": "TEXT",
    },
    "observations": {
        "measurement_id": "TEXT",
        "assay_id": "TEXT",
        "subject_participant_id": "TEXT",
        "quantity": "TEXT",
        "value": "REAL",
        "unit": "TEXT",
        "relation": "TEXT",
        "lower_bound": "REAL",
        "upper_bound": "REAL",
        "uncertainty_kind": "TEXT",
        "uncertainty_value": "REAL",
        "ci_lower": "REAL",
        "ci_upper": "REAL",
        "ci_level": "REAL",
        "replicate_count": "INTEGER",
        "time_s": "REAL",
        "replicate_id": "TEXT",
        "measurement_scope": "TEXT",
        "equivalence_group": "TEXT",
        "source_record_id": "TEXT",
        "raw_value": "TEXT",
        "raw_unit": "TEXT",
        "derivation": "TEXT",
    },
    "contexts": {
        "context_id": "TEXT",
        "assay_id": "TEXT",
        "substrate_participant_id": "TEXT",
        "enzyme_participant_id": "TEXT",
        "cut_after_residue": "INTEGER",
        "coordinate_basis": "TEXT",
        "residues_json": "TEXT",
        "position_status_json": "TEXT",
        "site_evidence": "TEXT",
        "source_record_id": "TEXT",
    },
    "observation_contexts": {
        "measurement_id": "TEXT",
        "context_id": "TEXT",
        "attribution": "TEXT",
    },
    "observation_inputs": {
        "input_id": "TEXT",
        "measurement_id": "TEXT",
        "role": "TEXT",
        "value": "REAL",
        "unit": "TEXT",
        "source_record_id": "TEXT",
        "source_field": "TEXT",
    },
    "specificity_profiles": {
        "profile_id": "TEXT",
        "dataset_index": "INTEGER",
        "folder": "TEXT",
        "merops_code": "TEXT",
        "enzyme_id": "TEXT",
        "n_cleavages": "REAL",
        "source_record_id": "TEXT",
    },
    "profile_cells": {
        "profile_id": "TEXT",
        "position_index": "INTEGER",
        "residue_index": "INTEGER",
        "residue": "TEXT",
        "count": "REAL",
        "log_probability": "REAL",
    },
    "context_features": {
        "feature_id": "TEXT",
        "context_id": "TEXT",
        "profile_id": "TEXT",
        "quantity": "TEXT",
        "value": "REAL",
        "unit": "TEXT",
        "n_observed_positions": "INTEGER",
        "n_scored_positions": "INTEGER",
        "status": "TEXT",
        "method": "TEXT",
        "source_record_id": "TEXT",
    },
}
PRIMARY_KEYS = {name: [next(iter(columns))] for name, columns in TABLES.items()}
PRIMARY_KEYS.update(
    observation_contexts=["measurement_id", "context_id"],
    profile_cells=["profile_id", "position_index", "residue_index"],
)
FOREIGN_KEYS = {
    "assays": {
        "condition_id": ("condition_sets", "condition_id"),
        "source_record_id": ("source_records", "record_id"),
    },
    "substrates": {"source_record_id": ("source_records", "record_id")},
    "enzymes": {"source_record_id": ("source_records", "record_id")},
    "participants": {
        "assay_id": ("assays", "assay_id"),
        "substrate_id": ("substrates", "substrate_id"),
        "enzyme_id": ("enzymes", "enzyme_id"),
    },
    "observations": {
        "assay_id": ("assays", "assay_id"),
        "subject_participant_id": ("participants", "participant_id"),
        "source_record_id": ("source_records", "record_id"),
    },
    "contexts": {
        "assay_id": ("assays", "assay_id"),
        "substrate_participant_id": ("participants", "participant_id"),
        "enzyme_participant_id": ("participants", "participant_id"),
        "source_record_id": ("source_records", "record_id"),
    },
    "observation_contexts": {
        "measurement_id": ("observations", "measurement_id"),
        "context_id": ("contexts", "context_id"),
    },
    "observation_inputs": {
        "measurement_id": ("observations", "measurement_id"),
        "source_record_id": ("source_records", "record_id"),
    },
    "specificity_profiles": {
        "enzyme_id": ("enzymes", "enzyme_id"),
        "source_record_id": ("source_records", "record_id"),
    },
    "profile_cells": {"profile_id": ("specificity_profiles", "profile_id")},
    "context_features": {
        "context_id": ("contexts", "context_id"),
        "profile_id": ("specificity_profiles", "profile_id"),
        "source_record_id": ("source_records", "record_id"),
    },
}


def plain(value):
    """Convert source/numpy values to JSON-compatible objects without NaN.

    :param value: A source scalar or nested record.
    :return: A serializable scalar or nested record with null missing values.
    """
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if type(value).__name__ in {"NAType", "NaTType"}:
        return None
    if type(value).__name__ == "Decimal":
        return float(value)
    return value


def canonical(value):
    """Encode a deterministic JSON value for hashes and typed JSON columns.

    :param value: Serializable source description.
    :return: Canonical JSON string.
    :rtype: str
    """
    return json.dumps(
        plain(value),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        default=str,
    )


def stable_id(kind, *parts):
    """Identify a scoped description or source record, never numeric coincidence.

    :param str kind: Entity namespace.
    :param parts: Explicit source identities or complete description fields.
    :return: Namespaced deterministic SHA-256 identifier.
    :rtype: str
    """
    return kind + ":" + hashlib.sha256(canonical(parts).encode()).hexdigest()[:32]
