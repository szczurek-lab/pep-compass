"""Condition protocols consume and export the compact public contract."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from proteolysis_data_manager.protocols.half_life.conditions import condition_status
from proteolysis_data_manager.integration.half_life.schema import CLEAN_COLUMNS, SCHEMA_VERSION, read_parquet, source_mask, typed, write_csv
from proteolysis_data_manager.utils.storage import protocol_manifest, source_csv


def load_half_life_data(data_root, source=None):
    path = Path(data_root) / 'normalized/observations.parquet'
    manifest = json.loads((Path(data_root) / 'manifests/normalization_manifest.json').read_text())
    if (manifest['schema_version'] != SCHEMA_VERSION or
            manifest['normalized_parquet_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError('Normalized Parquet changed or uses a stale schema; rerun normalization')
    frame = read_parquet(path)
    if source is None:
        return frame
    if source not in {s for sources in frame.source for s in sources}:
        raise ValueError(f'Unknown half-life source: {source}')
    return frame.loc[source_mask(frame, source)].copy()


def prepare_protocol(frame, config):
    """Add transient decisions. Assumptions never overwrite reported pH/T."""
    frame = typed(frame)
    selected = pd.Series(True, index=frame.index)
    for field, value in config['group'].items():
        selected &= frame[field].eq(value).fillna(False)
    rows = frame.loc[selected].copy()
    for field in ['ph', 'temperature_c']:
        rows[field + '_effective'] = rows[field].fillna(config['defaults'][field])
        rows[field + '_origin'] = np.where(rows[field].notna(), 'reported', 'assumed_reference')
    bounds = {field + '_effective': interval for field, interval in config['acceptance_bounds'].items()}
    rows['condition_status'] = condition_status(rows, bounds)
    rows['aggregation_eligible'] = rows.condition_status.eq('accepted')
    rows['peptide_per_serum_um'] = rows.peptide_concentration_um.div(rows.matrix_fraction.where(rows.matrix_fraction.gt(0)))
    rows['label_availability'] = np.where(rows.half_life_seconds.notna(), 'numeric',
                                          np.where(rows.half_life_class.notna(), 'class_only', 'unresolved'))
    return rows


def export_protocol(rows, data_root, source, config):
    """Export all accepted labels, including missing numerical half-lives."""
    group = config['group']; environment = group['species'] + '_' + group['matrix']
    root = Path(data_root)
    retained = typed(rows.loc[rows.aggregation_eligible, CLEAN_COLUMNS])
    if len(retained) and not source_mask(retained, source).all():
        raise ValueError('Protocol source does not match source membership')
    path = source_csv(root, environment, source)
    write_csv(retained, path)
    normalization = root / 'manifests/normalization_manifest.json'
    manifest = {
        'schema_version': SCHEMA_VERSION, 'source': source, 'environment': environment, 'configuration': config,
        'selected_group_rows': len(rows), 'retained_rows': len(retained),
        'numeric_rows': int(retained.half_life_seconds.notna().sum()),
        'class_only_rows': int((retained.half_life_seconds.isna() & retained.half_life_class.notna()).sum()),
        'unresolved_rows': int((retained.half_life_seconds.isna() & retained.half_life_class.isna()).sum()),
        'condition_rejected_rows': int((~rows.aggregation_eligible).sum()),
        'source_csv': str(path.relative_to(root)), 'source_csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'normalization_manifest_sha256': hashlib.sha256(normalization.read_bytes()).hexdigest(),
        'protocol_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'precision_policy': 'Reported CI/SD only; missing remains null. No uncertainty estimation.',
    }
    manifest_path = protocol_manifest(root, environment, source)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    numeric = retained.loc[retained.half_life_seconds.notna()].copy()
    classes = retained.loc[retained.half_life_seconds.isna() & retained.half_life_class.notna()].copy()
    return numeric, classes, path.parent
