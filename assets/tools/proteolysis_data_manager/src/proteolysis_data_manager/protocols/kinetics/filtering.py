"""One reviewable BRENDA protocol; preserve numeric and incomplete observations."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from proteolysis_data_manager.utils.io import sha256


def evaluate_protocol(dataset, config):
    dataset.validate()
    allowed = {'condition_key', 'enzyme_species', 'experiment_filters', 'acceptance_bounds',
               'missing_condition_policy', 'references', 'description'}
    if set(config) - allowed:
        raise ValueError('Unknown protocol configuration field')
    if not re.fullmatch(r'[a-z][a-z0-9_]*', config['condition_key']):
        raise ValueError('Invalid condition key')
    if config['missing_condition_policy'] not in {'retain', 'reject'}:
        raise ValueError('Unknown missing-condition policy')
    if set(config['experiment_filters']) - {'species', 'matrix', 'in_vivo_in_vitro'}:
        raise ValueError('Unknown experimental group filter')
    if set(config['acceptance_bounds']) - {'ph', 'temperature_c'}:
        raise ValueError('Only pH and temperature are acceptance bounds at this stage')
    for bounds in config['acceptance_bounds'].values():
        if bounds is not None and (len(bounds) != 2 or bounds[0] > bounds[1]):
            raise ValueError('Invalid acceptance bounds')
    rows = dataset.measurements_view()
    rows['exclusion_reasons'] = [[] for _ in range(len(rows))]
    target = config.get('enzyme_species')
    links = dataset['experiment_enzymes_relation']
    known = links.merge(dataset['enzymes'][['enzyme_id', 'enzyme_species']], on='enzyme_id', validate='many_to_one')
    complete_species = known.groupby('experiment_id').enzyme_species.agg(lambda v: bool(v.notna().all()))
    for idx, row in rows.iterrows():
        reasons = []
        if target and (row.enzyme_species != [target] or not complete_species.get(row.experiment_id, False)):
            reasons.append('enzyme_species_not_uniquely_target')
        for field, value in config['experiment_filters'].items():
            if value is not None and (pd.isna(row[field]) or row[field] != value):
                reasons.append(f'not_documented_{field}')
        for field, bounds in config['acceptance_bounds'].items():
            if bounds is None:
                continue
            if pd.isna(row[field]):
                if config['missing_condition_policy'] == 'reject':
                    reasons.append(f'missing_{field}')
            elif not bounds[0] <= row[field] <= bounds[1]:
                reasons.append(f'outside_{field}')
        rows.at[idx, 'exclusion_reasons'] = reasons
    rows['accepted'] = rows.exclusion_reasons.map(len).eq(0)
    rows['numeric_model_eligible'] = rows.accepted & rows.value_numeric.gt(0) & rows.value_relation.eq('eq')
    return rows


def export_protocol(dataset, rows, root, config):
    """Write a referentially complete subset, retaining all accepted label kinds."""
    expected = evaluate_protocol(dataset, config)
    if (list(rows.measurement_id) != list(expected.measurement_id) or
            not rows.accepted.equals(expected.accepted)):
        raise ValueError('Protocol decisions do not match the current dataset/configuration')
    accepted = rows.loc[rows.accepted]
    subset = dataset.subset_experiments(accepted.experiment_id)
    if set(subset['kinetic_measurements'].measurement_id) != set(accepted.measurement_id):
        raise ValueError('Selection would split an experiment context; define the policy explicitly')
    root = Path(root)
    folder = root / 'processed' / config['condition_key'] / 'sources/brenda'
    metadata = {
        'source': 'brenda', 'condition': config['condition_key'], 'configuration': config,
        'normalization_manifest_sha256': sha256(root / 'manifests/normalization_brenda.json'),
        'protocol_sha256': sha256(Path(__file__)),
        'source_measurements': len(rows), 'accepted_measurements': len(accepted),
        'accepted_numeric': int(accepted.value_numeric.notna().sum()),
        'accepted_non_numeric': int(accepted.value_numeric.isna().sum()),
        'numeric_model_eligible': int(accepted.numeric_model_eligible.sum()),
        'rejected_measurements': int((~rows.accepted).sum()),
        'uncertainty_policy': 'Reported only; no imputation or CI estimation.',
    }
    manifest = subset.write(folder, root / 'manifests' / f'protocol_{config["condition_key"]}_brenda.json', metadata=metadata)
    return subset, manifest
