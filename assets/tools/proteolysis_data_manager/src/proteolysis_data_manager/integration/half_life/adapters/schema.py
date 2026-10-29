"""Shared peptide and experiment contracts for every half-life adapter."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np
import pandas as pd

from proteolysis_data_manager.integration.half_life.adapters.common import MISSING, condition_frame, normalized_sequence
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

SCHEMA_VERSION = '1.0'
SEQUENCE_COLUMNS = [
    'sequence_id', 'sequence', 'chirality', 'topology',
    'n_terminal_modification', 'c_terminal_modification', 'other_modifications',
]
EXPERIMENT_COLUMNS = [
    'experiment_id', 'sequence_id', 'source', 'source_record_id',
    'publication_id', 'source_locator', 'species', 'matrix', 'matrix_fraction',
    'in_vivo_in_vitro', 'assay', 'temperature_c', 'ph', 'peptide_concentration_um',
    'measurement_kind', 'relation', 'half_life_seconds', 'lower_bound_seconds',
    'upper_bound_seconds', 'ci_lower_seconds', 'ci_upper_seconds', 'ci_level',
    'uncertainty_kind', 'uncertainty_value_seconds', 'replicate_count',
]
FLOAT_COLUMNS = {
    'matrix_fraction', 'temperature_c', 'ph', 'peptide_concentration_um',
    'half_life_seconds', 'lower_bound_seconds', 'upper_bound_seconds',
    'ci_lower_seconds', 'ci_upper_seconds', 'ci_level', 'uncertainty_value_seconds',
}

def _text(value):
    if value is None or pd.isna(value):
        return None
    value = re.sub(r'\s+', ' ', str(value)).strip()
    return None if value.lower() in MISSING | {'unknown'} else value


def typed_frame(records, columns):
    frame = pd.DataFrame(records, columns=columns)
    for column in columns:
        if column in FLOAT_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors='raise').astype('Float64')
        elif column == 'replicate_count':
            frame[column] = pd.to_numeric(frame[column], errors='raise').astype('Int64')
        else:
            frame[column] = frame[column].astype('string')
    return frame


def sequence_record(record):
    """Build a stable identity from the complete reported peptide description.

    :param dict record: Source adapter inventory record.
    :return: The seven peptide contract fields, including a SHA-256 key.
    :rtype: dict

    REMARK: Sequence case is retained because lowercase residues can encode D
    stereochemistry. Missing chemistry never matches an explicit modification
    or an explicit unmodified value. Equal incomplete descriptions share an ID;
    this does not assert that their undocumented chemistry is identical.
    """
    raw_sequence = record.get('sequence_raw', record.get('sequence', ''))
    sequence = normalized_sequence(raw_sequence, preserve_case=True)
    peptide = {'sequence': sequence}
    for field in SEQUENCE_COLUMNS[2:]:
        value = _text(record.get('chemical_modifications' if field == 'other_modifications' else field))
        peptide[field] = value.lower() if value is not None else None
    if peptide['chirality'] == 'mix':
        peptide['chirality'] = 'mixed'
    payload = json.dumps(peptide, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return {'sequence_id': hashlib.sha256(payload.encode()).hexdigest(), **peptide}


@dataclass
class HalfTimeDataset:
    """Typed peptide registry and source observations linked by ``sequence_id``.

    :param pandas.DataFrame sequences_half_time: Seven-field peptide registry.
    :param pandas.DataFrame half_time_experiments: Source experimental records.
    """

    sequences_half_time: pd.DataFrame
    half_time_experiments: pd.DataFrame

    def validate(self):
        """Check schemas, primary/foreign keys and measurement semantics.

        :return: This validated dataset.
        :rtype: HalfTimeDataset
        :raises ValueError: If a key, field, numeric value or CI is invalid.
        """
        sequences, experiments = self.sequences_half_time, self.half_time_experiments
        for frame, columns, key in [
            (sequences, SEQUENCE_COLUMNS, 'sequence_id'),
            (experiments, EXPERIMENT_COLUMNS, 'experiment_id'),
        ]:
            if list(frame.columns) != columns:
                raise ValueError(f'Unexpected {key} contract columns')
            if frame[key].isna().any() or frame[key].eq('').any() or frame[key].duplicated().any():
                raise ValueError(f'Missing or duplicate {key}')
        if experiments.sequence_id.isna().any() or not experiments.sequence_id.isin(sequences.sequence_id).all():
            raise ValueError('Experiment sequence_id has no peptide record')
        for field in ['source', 'source_record_id', 'measurement_kind', 'relation']:
            if experiments[field].isna().any() or experiments[field].eq('').any():
                raise ValueError(f'Missing experiment {field}')
        if not experiments.relation.isin(['eq', 'lt', 'le', 'gt', 'ge', 'approx', 'interval', 'unknown']).all():
            raise ValueError('Invalid measurement relation')
        # Validate units and reported numerical ranges without imputing values.
        for field in FLOAT_COLUMNS | {'replicate_count'}:
            values = experiments[field].dropna().to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite {field}')
            if field not in {'temperature_c', 'ph'} and (values < 0).any():
                raise ValueError(f'Negative {field}')
        for field in ['half_life_seconds', 'replicate_count']:
            if experiments[field].dropna().le(0).any():
                raise ValueError(f'Nonpositive {field}')
        if not experiments.matrix_fraction.dropna().between(0, 1).all():
            raise ValueError('matrix_fraction must be in [0, 1]')
        lower, upper, level = [experiments[field] for field in ['ci_lower_seconds', 'ci_upper_seconds', 'ci_level']]
        if (lower.isna() != upper.isna()).any() or (level.notna() & lower.isna()).any():
            raise ValueError('Incomplete reported CI')
        if (upper < lower).any() or ((level <= 0) | (level >= 1)).any():
            raise ValueError('Invalid reported CI')
        if (experiments.upper_bound_seconds < experiments.lower_bound_seconds).any():
            raise ValueError('Invalid half-life interval')
        nonpoint = experiments.relation.isin(['lt', 'le', 'gt', 'ge', 'interval', 'unknown'])
        if experiments.loc[nonpoint, 'half_life_seconds'].notna().any():
            raise ValueError('A censored or unresolved observation cannot have a point half-life')
        not_half_life = experiments.measurement_kind.ne('half_life')
        if experiments.loc[not_half_life, 'half_life_seconds'].notna().any():
            raise ValueError('A non-half-life readout cannot have a point half-life')
        return self


def adapt_records(records):
    """Split source records into peptide identity and experimental observations.

    :param iterable records: Detailed adapter records with original source IDs.
    :return: Typed tables with deduplicated descriptions and separate observations.
    :rtype: HalfTimeDataset
    :raises ValueError: If source IDs, measurements or output relationships fail.

    Approximate values retain ``relation='approx'``. Ranges, bounds and CI do not
    supply invented point estimates. Non-half-life readouts remain unlabelled;
    their original descriptions are available through the source record ID.
    """
    # Reuse explicit unit/condition parsing, without applying protocol filters.

    records = list(records)
    if not records:
        return HalfTimeDataset(typed_frame([], SEQUENCE_COLUMNS), typed_frame([], EXPERIMENT_COLUMNS)).validate()
    inventory = condition_frame(pd.DataFrame(records))
    peptides, experiments = {}, []
    for record in inventory.to_dict('records'):
        peptide = sequence_record(record)
        peptides[peptide['sequence_id']] = peptide
        experiment = {field: record.get(field) for field in EXPERIMENT_COLUMNS}
        experiment.update(
            experiment_id=record['observation_id'], sequence_id=peptide['sequence_id'],
            source_record_id=str(record['source_record_id']),
            source_locator=_text(record.get('source_locator')) or
                f"{record.get('source_filename', 'source record')}#{record['source_record_id']}",
            ci_level=record.get('ci_confidence_level'), replicate_count=record.get('n_replicates'),
        )
        # Retain the reported point or approximation, never a range/CI midpoint.
        if record.get('relation') == 'approx':
            experiment['half_life_seconds'] = record.get('approximate_half_life_seconds')
        uncertainty = record.get('uncertainty_kind', 'not_reported')
        error = record.get('reported_error_seconds')
        for field, kind in [('half_life_sd_seconds', 'sd'), ('half_life_se_seconds', 'se')]:
            if record.get(field) is not None and pd.notna(record[field]):
                uncertainty, error = kind, record[field]
                break
        experiment['uncertainty_kind'] = uncertainty
        experiment['uncertainty_value_seconds'] = error
        for field in set(EXPERIMENT_COLUMNS) - FLOAT_COLUMNS - {'replicate_count'}:
            if field not in {'relation', 'measurement_kind'}:
                experiment[field] = _text(experiment[field])
        experiments.append(experiment)
    dataset = HalfTimeDataset(
        typed_frame(sorted(peptides.values(), key=lambda r: r['sequence_id']), SEQUENCE_COLUMNS),
        typed_frame(sorted(experiments, key=lambda r: r['experiment_id']), EXPERIMENT_COLUMNS),
    )
    logger.debug('Adapted %s observations to %s peptide descriptions', len(experiments), len(peptides))
    return dataset.validate()
