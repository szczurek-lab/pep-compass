"""Condition-wise union of compact protocol CSVs, preserving verified lineage."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
from proteolysis_data_manager.integration.half_life.schema import (CLEAN_COLUMNS, SCHEMA_VERSION, VALUE_COLUMNS, canonical_json, plain,
                     read_csv, row_key, source_mask, typed, values_key, write_csv)
from proteolysis_data_manager.utils.storage import protocol_manifest, aggregation_manifest
from proteolysis_data_manager.protocols.half_life import filtering

# Final aggregation has exactly the same public columns as normalization.
AGGREGATE_COLUMNS = []


def merge_observations(frame):
    """Merge matching measurements only when a source reference is shared.

    Different experiments with equal sequence and label remain separate. Input
    protocols select the same already enriched normalized observation, so shared
    rows normally have identical values and complete contributor references.
    """
    frame = typed(frame)
    groups = {}
    for record in frame.to_dict('records'):
        groups.setdefault(values_key(record), []).append(plain(record))
    output, lineage = [], []
    for records in groups.values():
        parent = list(range(len(records)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        seen = {}
        for i, record in enumerate(records):
            for ref in record['source_record_id']:
                key = canonical_json(ref)
                if key in seen:
                    parent[find(i)] = find(seen[key])
                seen[key] = i
        components = {}
        for i, record in enumerate(records):
            components.setdefault(find(i), []).append(record)
        for members in components.values():
            row = {field: members[0][field] for field in VALUE_COLUMNS}
            row['source'] = sorted({s for member in members for s in member['source']})
            refs = {canonical_json(ref): ref for member in members for ref in member['source_record_id']}
            row['source_record_id'] = [refs[k] for k in sorted(refs)]
            output.append(row)
            for ref in row['source_record_id']:
                lineage.append({'normalized_row_key': row_key(row), **ref})
    return typed(pd.DataFrame(output, columns=CLEAN_COLUMNS)), pd.DataFrame(lineage, columns=['normalized_row_key', 'source', 'id'])


def possible_duplicates(frame):
    """Sequence/context coincidences are audited, never treated as identity."""
    fields = ['sequence', 'species', 'matrix', 'in_vivo_in_vitro', 'half_life_seconds']
    points = frame.loc[frame.half_life_seconds.notna()]
    return points.loc[points.duplicated(fields, keep=False)].copy()


def aggregate_environment(data_root, environment, sources):
    root = Path(data_root)
    tables, receipts, policies = [], [], []
    for source in sources:
        mp = protocol_manifest(root, environment, source)
        manifest = json.loads(mp.read_text())
        path = root / manifest['source_csv']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest['source_csv_sha256']:
            raise ValueError(f'Source CSV changed: {path}')
        if manifest['schema_version'] != SCHEMA_VERSION:
            raise ValueError(f'Stale source schema: {mp}')
        current = hashlib.sha256((root / 'manifests/normalization_manifest.json').read_bytes()).hexdigest()
        if manifest['normalization_manifest_sha256'] != current:
            raise ValueError(f'Stale normalized inventory: {mp}')
        if manifest['protocol_sha256'] != hashlib.sha256(Path(filtering.__file__).read_bytes()).hexdigest():
            raise ValueError(f'Stale source protocol: {mp}')
        table = read_csv(path)
        if not source_mask(table, source).all():
            raise ValueError(f'Incorrect source membership: {path}')
        config = manifest['configuration']
        if config['group']['species'] + '_' + config['group']['matrix'] != environment:
            raise ValueError(f'Incorrect condition: {mp}')
        for field, value in config['group'].items():
            if not table[field].eq(value).fillna(False).all():
                raise ValueError(f'Incorrect condition values: {path}')
        from proteolysis_data_manager.protocols.half_life.filtering import prepare_protocol
        checked = prepare_protocol(table, config)
        if len(checked) != len(table) or not checked.aggregation_eligible.all():
            raise ValueError(f'Rejected observations in protocol export: {path}')
        policies.append({k: config[k] for k in ['group', 'defaults', 'acceptance_bounds']})
        tables.append(table)
        receipts.append({'source': source, 'csv': str(path.relative_to(root)), 'sha256': digest, 'source_rows': len(table)})
    if not sources or len({canonical_json(p) for p in policies}) != 1:
        raise ValueError('Source protocols must use one identical acceptance policy')
    incoming = typed(pd.concat(tables, ignore_index=True))
    merged, lineage = merge_observations(incoming)
    candidates = possible_duplicates(merged)
    path = root / 'processed' / environment / (environment + '.csv')
    write_csv(merged, path)
    manifest = {'schema_version': SCHEMA_VERSION, 'environment': environment, 'inputs': receipts,
                'configuration': policies[0], 'input_rows': len(incoming), 'output_rows': len(merged),
                'merged_retained_rows': len(incoming) - len(merged), 'lineage_rows': len(lineage),
                'possible_duplicate_rows': len(candidates),
                'duplicate_policy': 'Shared source reference and identical scientific fields; independent coincidences retained.',
                'ci_policy': 'Reported only; missing remains null.',
                'output_csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    aggregation_manifest(root, environment).write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return incoming, merged, lineage, candidates, manifest


def source_accounting(data_root, inventory, incoming, merged, config, sources):
    """Show all source denominators and numeric/class-only/unresolved coverage."""
    group = config['group']; environment = group['species'] + '_' + group['matrix']
    reference = incoming.loc[source_mask(incoming, 'peplife2')]
    reference_sequences = set(reference.sequence.dropna())
    coverage, labels, overlap, missing = [], [], [], []
    normalization = json.loads((Path(data_root) / 'manifests/normalization_manifest.json').read_text())
    for source in sources:
        full = inventory.loc[source_mask(inventory, source)]
        condition = full.loc[full.species.eq(group['species']) & full.matrix.eq(group['matrix'])]
        selected = condition.loc[condition.in_vivo_in_vitro.eq(group['in_vivo_in_vitro'])]
        data = incoming.loc[source_mask(incoming, source)]
        # A shared normalized row is exported by each contributing protocol.
        data = data.loc[~data.apply(row_key, axis=1).duplicated()].copy()
        receipt = json.loads(protocol_manifest(data_root, environment, source).read_text())
        coverage.append({'source': source,
                         'raw_source_representations': normalization.get('raw_source_rows', {}).get(source),
                         'normalized_measurements': len(full), 'condition_records': len(condition),
                         'selected_assay_mode': len(selected),
                         'other_known_mode': int((condition.in_vivo_in_vitro.notna() & condition.in_vivo_in_vitro.ne(group['in_vivo_in_vitro'])).sum()),
                         'unknown_assay_mode': int(condition.in_vivo_in_vitro.isna().sum()),
                         'outside_ph_temperature': receipt['condition_rejected_rows'],
                         'exported_source_records': len(data)})
        point = data.half_life_seconds.notna(); cls = ~point & data.half_life_class.notna()
        labels.append({'source': source, 'exported_records': len(data), 'numeric': int(point.sum()),
                       'class_only': int(cls.sum()), 'unresolved': int((~point & ~cls).sum()),
                       'without_numeric_half_life': int((~point).sum())})
        shared_seq = data.sequence.isin(reference_sequences)
        overlap.append({'source': source, 'comparison': 'reference' if source == 'peplife2' else 'against PEPlife2',
                        'shared_sequences': data.loc[shared_seq, 'sequence'].nunique(),
                        'shared_sequence_records': int(shared_seq.sum()),
                        'verified_same_measurements': None if source == 'peplife2' else int(source_mask(data, 'peplife2').sum()),
                        'new_sequences': data.loc[~shared_seq, 'sequence'].nunique()})
        for field in ['half_life_seconds', 'half_life_class', 'ph', 'temperature_c', 'matrix_fraction',
                      'peptide_concentration_um', 'sd_seconds', 'ci_lower_seconds', 'ci_upper_seconds', 'ci_level', 'replicate_count']:
            missing.append({'source': source, 'parameter': field, 'documented': int(data[field].notna().sum()),
                            'missing': int(data[field].isna().sum()), 'denominator': len(data)})
    return {'coverage': pd.DataFrame(coverage), 'label_types': pd.DataFrame(labels),
            'peplife2_overlap': pd.DataFrame(overlap), 'missingness': pd.DataFrame(missing)}
