"""Build and validate a common dataset while keeping source experiment identities."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from proteolysis_data_manager.integration.merge.contract.schema import (
    FOREIGN_KEYS,
    POSITIONS,
    PRIMARY_KEYS,
    QUANTITIES,
    TABLES,
)
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class UnifiedDataset:
    """Common tables, quantity definitions and cardinality-preserving public views.

    :param tables: Frames following the versioned schema.
    :param metadata: Build metadata, stored inside SQLite rather than sidecar files.
    """

    def __init__(self, tables, metadata=None):
        self.tables = tables
        self.metadata = metadata or {}

    def to_frame(
        self,
        *,
        source=None,
        quantities=None,
        merops_dataset_index=34,
        include_probabilities=True,
    ):
        """Read the common wide feature table with numeric parameters and cut windows.

        :param source: Optional source database selector.
        :param quantities: Optional measurement-name row selector.
        :param int merops_dataset_index: Explicit MEROPS population for reference scores.
        :param bool include_probabilities: Add empirical positional probabilities.
        :return: Wide parameter/percentage/probability table with original associations.
        """
        from proteolysis_data_manager.integration.merge.views.wide import common_frame

        return common_frame(
            self,
            source=source,
            quantities=quantities,
            merops_dataset_index=merops_dataset_index,
            include_probabilities=include_probabilities,
        )

    def position_probabilities(self, *, frame=None):
        """Read normalized source/profile matrices with amino acids as columns.

        :param frame: Optional already-computed common wide table.
        :return: Source, enzyme, position, depth and 20 amino-acid probability columns.
        """
        from proteolysis_data_manager.integration.merge.features.frequencies import (
            position_probabilities,
        )

        return position_probabilities(self, frame=frame)

    def table(self, name):
        """Return one contract table.

        :param str name: Registered table name.
        :return: Table frame.
        """
        return self.tables[name]

    def validate(self):
        """Check primary/foreign keys, units, assay scope and positional semantics.

        :return: This dataset.
        :raises ValueError: If schema, identifiers, units or associations conflict.
        """
        if set(self.tables) != set(TABLES):
            raise ValueError("Incomplete unified dataset")
        for name, columns in TABLES.items():
            frame = self.tables[name]
            if list(frame.columns) != list(columns):
                raise ValueError(f"Incorrect schema: {name}")
            if (
                frame[PRIMARY_KEYS[name]].isna().any().any()
                or frame.duplicated(PRIMARY_KEYS[name]).any()
            ):
                raise ValueError(f"Missing/duplicate primary key: {name}")
            for column, kind in columns.items():
                if kind in {"REAL", "INTEGER"}:
                    values = pd.to_numeric(frame[column].dropna(), errors="raise")
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite {name}.{column}")
            for column, (target, key) in FOREIGN_KEYS.get(name, {}).items():
                if not frame[column].dropna().isin(self.tables[target][key]).all():
                    raise ValueError(f"Dangling {name}.{column}")
        participants = self.tables["participants"]
        if (participants.substrate_id.notna() == participants.enzyme_id.notna()).any():
            raise ValueError("A participant must identify exactly one entity")
        observations = self.tables["observations"]
        if not observations.quantity.isin(QUANTITIES).all():
            raise ValueError("Unknown quantity")
        if not observations.relation.isin(
            ["eq", "approx", "lt", "le", "gt", "ge", "interval", "unknown"]
        ).all():
            raise ValueError("Unknown relation")
        for quantity, (unit, signed, _) in QUANTITIES.items():
            frame = observations.loc[observations.quantity.eq(quantity)]
            if not (
                frame.unit.isna().all() if unit is None else frame.unit.eq(unit).all()
            ):
                raise ValueError(f"Incorrect {quantity} unit")
            if not signed and (frame.value.dropna() < 0).any():
                raise ValueError(f"Negative {quantity}")
        if (
            observations.loc[
                ~observations.relation.isin(["eq", "approx"]), "value"
            ].notna()
        ).any():
            raise ValueError(
                "Censored/unresolved observations cannot have invented point values"
            )
        if (observations.lower_bound > observations.upper_bound).fillna(False).any():
            raise ValueError("Reversed bounds")
        participant_assay = participants.set_index("participant_id").assay_id.to_dict()
        for row in observations.itertuples():
            if (
                row.subject_participant_id is not None
                and pd.notna(row.subject_participant_id)
                and participant_assay[row.subject_participant_id] != row.assay_id
            ):
                raise ValueError("Measured participant outside assay")
        contexts = self.tables["contexts"]
        for row in contexts.itertuples():
            residues, status = (
                json.loads(row.residues_json),
                json.loads(row.position_status_json),
            )
            if len(residues) != 8 or len(status) != 8:
                raise ValueError("Window must have eight positions")
            if any(
                s not in {"observed", "outside_substrate", "unresolved"} for s in status
            ):
                raise ValueError("Unknown position status")
            if any((r is None) != (s != "observed") for r, s in zip(residues, status)):
                raise ValueError("Residue/status mismatch")
            for participant in [
                row.substrate_participant_id,
                row.enzyme_participant_id,
            ]:
                if (
                    participant is not None
                    and pd.notna(participant)
                    and participant_assay[participant] != row.assay_id
                ):
                    raise ValueError("Context participant outside assay")
        assay_by_measurement = observations.set_index(
            "measurement_id"
        ).assay_id.to_dict()
        assay_by_context = contexts.set_index("context_id").assay_id.to_dict()
        for row in self.tables["observation_contexts"].itertuples():
            if (
                assay_by_measurement[row.measurement_id]
                != assay_by_context[row.context_id]
            ):
                raise ValueError("Cross-assay context link")
        # Check source marginal probabilities without recomputing their smoothing.
        cells = self.tables["profile_cells"]
        if not cells.empty:
            if (cells["count"] < 0).any() or (cells.log_probability > 0).any():
                raise ValueError("Invalid source profile values")
            totals = (
                np.exp(cells.log_probability)
                .groupby([cells.profile_id, cells.position_index])
                .sum()
            )
            # REMARK: Source matrices use float32; the bound covers rounding of 20 terms.
            if not np.allclose(totals, 1.0, rtol=0, atol=2e-6):
                raise ValueError("Unnormalized source profile")
            if not cells.groupby("profile_id").size().eq(160).all():
                raise ValueError("Incomplete source profile axes")
        # Keep score coverage and numerical meaning explicit for partial windows.
        features = self.tables["context_features"]
        allowed = {
            "merops_probability_sum",
            "merops_log_probability_sum",
            "merops_probability_product",
        }
        if not features.quantity.isin(allowed).all():
            raise ValueError("Unknown context feature")
        if (
            (features.n_scored_positions < 0)
            | (features.n_scored_positions > features.n_observed_positions)
            | (features.n_observed_positions > 8)
        ).any():
            raise ValueError("Invalid feature position coverage")
        for quantity in allowed:
            subset = features.loc[features.quantity.eq(quantity)]
            if not subset.unit.eq(QUANTITIES[quantity][0]).all():
                raise ValueError("Incorrect feature unit")
            if (
                quantity == "merops_probability_sum"
                and (
                    (subset.value < 0) | (subset.value > subset.n_scored_positions)
                ).any()
            ):
                raise ValueError("Invalid probability sum")
            if (
                quantity == "merops_probability_product"
                and ((subset.value < 0) | (subset.value > 1)).any()
            ):
                raise ValueError("Invalid probability product")
            if quantity == "merops_log_probability_sum" and (subset.value > 0).any():
                raise ValueError("Invalid log-probability sum")
        if (
            features.loc[
                ~features.status.isin(["complete_window", "partial_window"]), "value"
            ]
            .notna()
            .any()
        ):
            raise ValueError("Unresolved score has a point value")
        return self

    def measurements(self, *, source=None, quantity=None):
        """Return one row per measurement, with source and shared condition fields.

        :param source: Optional source-database selector.
        :param quantity: Optional registered quantity selector.
        :return: Frame; participants/sites never multiply numeric observations.
        """
        frame = self.tables["observations"].merge(
            self.tables["source_records"].drop(columns=["raw_json"]),
            left_on="source_record_id",
            right_on="record_id",
            validate="many_to_one",
        )
        frame = frame.merge(
            self.tables["assays"][["assay_id", "condition_id"]],
            on="assay_id",
            validate="many_to_one",
        ).merge(
            self.tables["condition_sets"], on="condition_id", validate="many_to_one"
        )
        if source is not None:
            frame = frame.loc[frame.source_db.eq(source)]
        if quantity is not None:
            frame = frame.loc[frame.quantity.eq(quantity)]
        return frame.reset_index(drop=True)

    def contexts(self, *, source=None):
        """Expose the same eight residue/status columns for every source.

        :param source: Optional source-database selector.
        :return: One row per context, with source identifiers and position fields.
        """
        frame = self.tables["contexts"].merge(
            self.tables["source_records"][
                [
                    "record_id",
                    "source_db",
                    "source_dataset",
                    "source_id",
                    "raw_path",
                    "locator",
                ]
            ],
            left_on="source_record_id",
            right_on="record_id",
            validate="many_to_one",
        )
        for i, position in enumerate(POSITIONS):
            frame[position] = frame.residues_json.map(
                lambda value, index=i: json.loads(value)[index]
            )
            frame[position + "_status"] = frame.position_status_json.map(
                lambda value, index=i: json.loads(value)[index]
            )
        if source is not None:
            frame = frame.loc[frame.source_db.eq(source)]
        return frame.reset_index(drop=True)

    def features(self):
        """Return derived MEROPS features with both profile and context origins.

        :return: One row per named context/profile quantity; no empirical value duplication.
        """
        frame = self.tables["context_features"].merge(
            self.tables["contexts"][["context_id", "assay_id"]],
            on="context_id",
            validate="many_to_one",
        )
        context_sources = self.tables["source_records"][
            ["record_id", "source_db", "source_dataset"]
        ].rename(
            columns={
                "source_db": "context_source_db",
                "source_dataset": "context_source_dataset",
            }
        )
        frame = frame.merge(
            context_sources,
            left_on="source_record_id",
            right_on="record_id",
            validate="many_to_one",
        )
        profiles = self.tables["specificity_profiles"][
            ["profile_id", "dataset_index", "merops_code", "source_record_id"]
        ].rename(columns={"source_record_id": "profile_source_record_id"})
        frame = frame.merge(
            profiles, on="profile_id", how="left", validate="many_to_one"
        )
        frame["source_db"] = "merops"
        frame["source_dataset"] = (
            frame["dataset_index"].astype("Int64").astype("string")
        )
        return frame

    def profile_values(self):
        """Expose original count/log-probability cells through registered quantities.

        :return: Long frame with the same quantity/value/unit/source naming as measurements.
        """
        frame = self.tables["profile_cells"].merge(
            self.tables["specificity_profiles"][["profile_id", "source_record_id"]],
            on="profile_id",
            validate="many_to_one",
        )
        frame = frame.merge(
            self.tables["source_records"][["record_id", "source_db", "source_dataset"]],
            left_on="source_record_id",
            right_on="record_id",
            validate="many_to_one",
        )
        result = frame.melt(
            id_vars=[c for c in frame if c not in {"count", "log_probability"}],
            value_vars=["count", "log_probability"],
            var_name="quantity",
            value_name="value",
        )
        result["quantity"] = result["quantity"].map(
            {"count": "merops_count", "log_probability": "merops_log_probability"}
        )
        result["unit"] = result["quantity"].map(lambda q: QUANTITIES[q][0])
        return result

    def values(self, *, include_profiles=False):
        """Read all scalar quantities with common names and explicit value origins.

        :param bool include_profiles: Include original marginal profile cells.
        :return: Long frame; empirical measurements are never exploded over sites.
        """
        measurements = self.measurements().copy()
        links = (
            self.tables["observation_contexts"]
            .groupby("measurement_id")
            .context_id.agg(list)
            .to_dict()
        )
        measurements["context_ids"] = measurements.measurement_id.map(
            lambda key: links.get(key, [])
        )
        measurements["value_origin"] = "measurement"
        features = self.features().copy()
        features["value_origin"] = "derived_feature"
        frames = [measurements, features]
        if include_profiles:
            profiles = self.profile_values()
            profiles["value_origin"] = "source_profile_cell"
            frames.append(profiles)
        return pd.concat(frames, ignore_index=True, sort=False)

    def summary(self):
        """Summarize actual common quantities by source and input subset.

        :return: Counts and registered units for measurements, features and profiles.
        """
        outputs = []
        for origin, frame in [
            ("measurement", self.measurements()),
            ("derived_feature", self.features()),
            ("source_profile_cell", self.profile_values()),
        ]:
            if frame.empty:
                continue
            summary = (
                frame.groupby(
                    ["source_db", "source_dataset", "quantity", "unit"], dropna=False
                )
                .agg(observations=("value", "size"), numeric=("value", "count"))
                .reset_index()
            )
            summary["value_origin"] = origin
            outputs.append(summary)
        return (
            pd.concat(outputs, ignore_index=True)
            if outputs
            else pd.DataFrame(
                columns=[
                    "source_db",
                    "source_dataset",
                    "quantity",
                    "unit",
                    "observations",
                    "numeric",
                    "value_origin",
                ]
            )
        )
