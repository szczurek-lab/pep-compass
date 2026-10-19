"""One public half-life contract, from normalized Parquet to condition CSVs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from proteolysis_data_manager.integration.half_life.labels import point_class

SCHEMA_VERSION = '4.0'
CLEAN_COLUMNS = [
    'source', 'source_record_id', 'sequence', 'sequence_length', 'chirality', 'topology',
    'n_terminal_modification', 'c_terminal_modification', 'other_modifications',
    'species', 'matrix', 'matrix_fraction', 'temperature_c', 'ph', 'peptide_concentration_um',
    'half_life_seconds', 'sd_seconds', 'ci_lower_seconds', 'ci_upper_seconds', 'ci_level',
    'replicate_count', 'half_life_class', 'in_vivo_in_vitro',
]
LIST_COLUMNS = ['source', 'source_record_id']
INTEGER_COLUMNS = ['sequence_length', 'replicate_count']
FLOAT_COLUMNS = ['matrix_fraction', 'temperature_c', 'ph', 'peptide_concentration_um',
                 'half_life_seconds', 'sd_seconds', 'ci_lower_seconds', 'ci_upper_seconds', 'ci_level']
VALUE_COLUMNS = [c for c in CLEAN_COLUMNS if c not in LIST_COLUMNS]
ARROW_SCHEMA = pa.schema([
    pa.field(c, pa.list_(pa.string()) if c == 'source' else
             pa.list_(pa.struct([('source', pa.string()), ('id', pa.string())])) if c == 'source_record_id' else
             pa.int64() if c in INTEGER_COLUMNS else pa.float64() if c in FLOAT_COLUMNS else pa.string())
    for c in CLEAN_COLUMNS
], metadata={b'half_life_schema_version': SCHEMA_VERSION.encode()})


def plain(value):
    if isinstance(value, np.ndarray):
        return [plain(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [plain(x) for x in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if value is None or pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def canonical_json(value):
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def validate(frame):
    if list(frame.columns) != CLEAN_COLUMNS:
        raise ValueError('Half-life table must have the 23 contract columns, in contract order')
    for record in frame.to_dict('records'):
        sources, refs = plain(record['source']), plain(record['source_record_id'])
        if not sources or sources != sorted(set(sources)):
            raise ValueError('source must be a sorted, nonempty list of unique source names')
        if not refs or {r['source'] for r in refs} != set(sources):
            raise ValueError('Every source must have a source_record_id reference')
        if any(set(r) != {'source', 'id'} or not r['id'] for r in refs):
            raise ValueError('Invalid source record reference')
        label = plain(record['half_life_class'])
        if label not in {None, 'short', 'medium', 'long'}:
            raise ValueError('Invalid half_life_class')
        for field in FLOAT_COLUMNS + INTEGER_COLUMNS:
            value = plain(record[field])
            if value is not None and not np.isfinite(value):
                raise ValueError(f'Non-finite {field}')
        point = plain(record['half_life_seconds'])
        if point is not None and (point <= 0 or label != point_class(point)):
            raise ValueError('Positive numerical half-life must have its corresponding duration class')
        fraction = plain(record['matrix_fraction'])
        if fraction is not None and not 0 <= fraction <= 1:
            raise ValueError('matrix_fraction must be in [0, 1]')
        for field in ['sd_seconds', 'peptide_concentration_um']:
            value = plain(record[field])
            if value is not None and value < 0:
                raise ValueError(f'Negative {field}')
        for field in INTEGER_COLUMNS:
            value = plain(record[field])
            if value is not None and (value < 1 or value != int(value)):
                raise ValueError(f'Invalid {field}')
        lower, upper, level = [plain(record[f]) for f in ['ci_lower_seconds', 'ci_upper_seconds', 'ci_level']]
        if (lower is None) != (upper is None) or (level is not None and lower is None):
            raise ValueError('CI bounds must occur together; level requires reported bounds')
        if lower is not None and (lower < 0 or upper < lower or (level is not None and not 0 < level < 1)):
            raise ValueError('Invalid reported confidence interval')
    return frame


def typed(frame):
    if list(frame.columns) != CLEAN_COLUMNS:
        raise ValueError('Unexpected half-life columns/order')
    frame = frame[CLEAN_COLUMNS].copy()
    for field in LIST_COLUMNS:
        frame[field] = frame[field].map(plain)
    for field in FLOAT_COLUMNS:
        frame[field] = pd.to_numeric(frame[field], errors='raise').astype('Float64')
    for field in INTEGER_COLUMNS:
        frame[field] = pd.to_numeric(frame[field], errors='raise').astype('Int64')
    for field in set(VALUE_COLUMNS) - set(FLOAT_COLUMNS + INTEGER_COLUMNS):
        frame[field] = frame[field].astype('string')
    return validate(frame)


def write_parquet(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    frame = typed(frame)
    pq.write_table(pa.Table.from_pylist([plain(r) for r in frame.to_dict('records')], schema=ARROW_SCHEMA), path)


def read_parquet(path):
    table = pq.read_table(path)
    if (not table.schema.equals(ARROW_SCHEMA, check_metadata=False) or
            (table.schema.metadata or {}).get(b'half_life_schema_version') != SCHEMA_VERSION.encode()):
        raise ValueError('Unexpected normalized Parquet contract/version')
    return typed(pd.DataFrame(table.to_pylist(), columns=CLEAN_COLUMNS))


def write_csv(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    out = typed(frame).copy()
    for field in LIST_COLUMNS:
        out[field] = out[field].map(canonical_json)
    out.to_csv(path, index=False, na_rep='')


def read_csv(path):
    strings = {c: 'string' for c in CLEAN_COLUMNS if c not in FLOAT_COLUMNS + INTEGER_COLUMNS}
    frame = pd.read_csv(path, dtype=strings, keep_default_na=False, na_values=[''])
    for field in LIST_COLUMNS:
        frame[field] = frame[field].map(json.loads)
    return typed(frame)


def source_mask(frame, source):
    return frame.source.map(lambda sources: source in sources).astype(bool)


def values_key(record):
    return canonical_json({c: record[c] for c in VALUE_COLUMNS})


def row_key(record):
    return hashlib.sha256(canonical_json({c: record[c] for c in CLEAN_COLUMNS}).encode()).hexdigest()


def _nullable(value):
    value = plain(value)
    if isinstance(value, str) and value.strip().lower() in {'', 'unknown', 'n.a.', 'n.a', 'na', 'n/a', 'nan', 'not mentioned', 'not given', 'not available', 'not found'}:
        return None
    return value


def compact_record(record):
    aliases = {'sequence_length': 'length', 'other_modifications': 'chemical_modifications',
               'sd_seconds': 'half_life_sd_seconds', 'ci_level': 'ci_confidence_level', 'replicate_count': 'n_replicates',
               'half_life_class': 'duration_class'}
    out = {c: _nullable(record.get(aliases.get(c, c))) for c in VALUE_COLUMNS}
    out['source'] = [record['source']]
    # View and split-clause suffix retain distinct observations from one source ID.
    out['source_record_id'] = [{'source': record['source'], 'id': record['observation_id'].removeprefix(record['source'] + ':')}]
    if record.get('measurement_kind') != 'half_life' or record.get('relation') != 'eq':
        out['half_life_seconds'] = None
    # Only explicitly documented CI may supply a representative midpoint.
    if record.get('measurement_kind') == 'half_life' and out['half_life_seconds'] is None and record.get('ci_origin', 'not_reported') != 'not_reported' and all(out[f] is not None for f in ['ci_lower_seconds', 'ci_upper_seconds', 'ci_level']):
        out['half_life_seconds'] = (out['ci_lower_seconds'] + out['ci_upper_seconds']) / 2
    if out['half_life_seconds'] is not None:
        out['half_life_class'] = point_class(out['half_life_seconds'])
    return out


def normalize_inventory(inventory):
    """Consolidate verified aliases; enrich nulls only when known values agree.

    Sequence equality is never an identity key. Conflicts retain separate rows,
    even when authors link the same historical record. Return an audit in memory.
    """
    groups = {}
    for raw in inventory.to_dict('records'):
        key = raw.get('representation_key') or raw['observation_id']
        groups.setdefault(key, []).append((raw, compact_record(raw)))
    output, audit = [], []
    for key, members in groups.items():
        conflicts = [c for c in VALUE_COLUMNS if len({canonical_json(out[c]) for _, out in members if out[c] is not None}) > 1]
        if not conflicts:
            subgroups = [members]
        else:
            split = {}
            for raw, out in members:
                split.setdefault(values_key(out), []).append((raw, out))
            subgroups = list(split.values())
        for subset in subgroups:
            merged = {c: next((out[c] for _, out in subset if out[c] is not None), None) for c in VALUE_COLUMNS}
            merged['source'] = sorted({raw['source'] for raw, _ in subset})
            refs = {canonical_json(ref): ref for _, out in subset for ref in out['source_record_id']}
            merged['source_record_id'] = [refs[k] for k in sorted(refs)]
            output.append(merged)
            for raw, out in subset:
                enriched = [c for c in VALUE_COLUMNS if out[c] is None and merged[c] is not None]
                origins = {c: [r['observation_id'] for r, value in subset if value[c] is not None] for c in enriched}
                audit.append({'observation_id': raw['observation_id'], 'source': raw['source'],
                              'merged_sources': merged['source'], 'enriched_fields': enriched,
                              'field_sources': origins,
                              'conflicting_fields': conflicts, 'normalized_row_key': row_key(merged)})
    return typed(pd.DataFrame(output, columns=CLEAN_COLUMNS)), pd.DataFrame(audit)
