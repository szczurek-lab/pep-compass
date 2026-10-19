"""Auditable coverage, overlap, chemistry and measurement-context diagnostics."""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from proteolysis_data_manager.integration.half_life.normalization import known

FEATURES = ['length', 'reference_pi', 'reference_charge_ph7', 'reference_gravy',
            'reference_molecular_weight_da', 'reference_aromaticity', 'reference_nonpolar_fraction']
DOCUMENTATION_FIELDS = ['publication_id', 'assay', 'in_vivo_in_vitro', 'chirality', 'topology',
                        'n_terminal_modification', 'c_terminal_modification', 'chemical_modifications',
                        'temperature_c', 'ph', 'matrix_fraction', 'peptide_concentration_raw', 'n_replicates',
                        'ci_lower_seconds', 'ci_upper_seconds']


def documented(series):
    return series.notna() & series.map(lambda x: known(x) and str(x).lower() != 'unknown')


def coverage(frame):
    records = []
    for (source, view), rows in frame.groupby(['source', 'dataset_view'], sort=False):
        human = rows[rows.is_human_serum]
        records.append({'dataset': source + '/' + view, 'source': source, 'dataset_view': view,
                        'n_rows': len(rows), 'n_sequences': rows.sequence.nunique(),
                        'n_human_serum': len(human), 'n_human_serum_sequences': human.sequence.nunique(),
                        'n_human_serum_in_vitro': int(human.in_vivo_in_vitro.eq('in vitro').sum()),
                        'n_human_serum_exact_half_life': int(human.is_exact_positive_half_life.sum()),
                        'n_baseline_eligible_rows': int(human.baseline_eligible.sum()),
                        'n_integrated_baseline_representatives': int((human.baseline_eligible & ~human.is_redundant_representation).sum()),
                        'n_redundant_representations': int(rows.is_redundant_representation.sum()),
                        'n_unresolved_context': int(rows.species.eq('unknown').sum()),
                        'n_metadata_recovered': int(rows.metadata_origin.str.startswith('peplife2_').sum()),
                        'n_fraction_readouts': int(rows.measurement_kind.eq('remaining_fraction').sum()),
                        'n_censored_half_lives': int(rows.relation.isin(['lt', 'le', 'gt', 'ge', 'interval']).sum())})
    return pd.DataFrame(records)


def documentation(frame):
    records = []
    for (source, view), rows in frame.groupby(['source', 'dataset_view']):
        for scope, subset in [('all_source_rows', rows), ('human_serum', rows[rows.is_human_serum])]:
            for field in DOCUMENTATION_FIELDS:
                count = int(documented(subset[field]).sum())
                records.append({'dataset': source+'/'+view, 'scope': scope, 'field': field,
                                'n_documented': count, 'n_rows': len(subset),
                                'fraction_documented': count/len(subset) if len(subset) else np.nan,
                                'n_direct_or_curated': int((documented(subset[field]) & ~subset.metadata_origin.str.startswith('peplife2_')).sum())})
    return pd.DataFrame(records)


def overlaps(frame, *, human_serum=False):
    if human_serum: frame = frame[frame.is_human_serum]
    groups = {source+'/'+view: rows for (source, view), rows in frame.groupby(['source', 'dataset_view'])}
    records = []
    for first, second in combinations(groups, 2):
        a, b = groups[first], groups[second]
        sa, sb = set(a.sequence), set(b.sequence)
        ca = set(a.loc[a.chemical_identity_complete, 'chemical_identity_key'])
        cb = set(b.loc[b.chemical_identity_complete, 'chemical_identity_key'])
        records.append({'dataset_a': first, 'dataset_b': second,
                        'scope': 'human_serum' if human_serum else 'all_source_rows',
                        'n_sequences_a': len(sa), 'n_sequences_b': len(sb), 'sequence_intersection': len(sa & sb),
                        'sequence_jaccard': len(sa & sb)/len(sa | sb) if sa | sb else np.nan,
                        'chemistry_intersection_where_complete': len(ca & cb),
                        'n_chemistry_known_a': len(ca), 'n_chemistry_known_b': len(cb)})
    return pd.DataFrame(records)


def linked_labels(frame):
    """Compare author-label representations only where ID and chemistry matched."""
    peplife = frame[frame.source.eq('peplife2')].set_index('source_record_id')
    records = []
    for _, r in frame[frame.linked_peplife2_id.ne('') & ~frame.is_redundant_representation].iterrows():
        p = peplife.loc[r.linked_peplife2_id]
        seconds = r.half_life_seconds
        compatible = None
        same = False
        if pd.notna(seconds):
            if p.relation in {'eq', 'approx'} and pd.notna(p.half_life_seconds):
                same = p.relation == r.relation and np.isclose(seconds, p.half_life_seconds, rtol=1e-6)
                compatible = bool(np.isclose(seconds, p.half_life_seconds, rtol=0.05))
            elif p.relation in {'lt', 'le'}:
                compatible = bool(seconds < p.upper_bound_seconds if p.relation == 'lt' else seconds <= p.upper_bound_seconds)
            elif p.relation in {'gt', 'ge'}:
                compatible = bool(seconds > p.lower_bound_seconds if p.relation == 'gt' else seconds >= p.lower_bound_seconds)
        records.append({'observation_id': r.observation_id, 'peplife2_id': p.name,
                        'is_human_serum': r.is_human_serum,
                        'author_relation': r.relation, 'author_seconds': seconds, 'peplife2_relation': p.relation,
                        'peplife2_seconds': p.half_life_seconds, 'peplife2_lower': p.lower_bound_seconds,
                        'peplife2_upper': p.upper_bound_seconds, 'same_label': same,
                        'compatible_with_reported_label': compatible,
                        'note': 'Matched source ID, name and recorded chemistry; author label retained.'})
    return pd.DataFrame(records)


def measurement_spread(frame):
    """Describe between-record heterogeneity; never claim technical precision."""
    rows = frame[frame.is_human_serum & frame.is_exact_positive_half_life & frame.chemical_identity_complete & ~frame.is_redundant_representation]
    columns = ['chemical_identity_key', 'matrix', 'in_vivo_in_vitro', 'n_records', 'n_publications', 'n_sources', 'min_seconds', 'max_seconds', 'max_min_ratio', 'interpretation']
    records = []
    for key, g in rows.groupby(['chemical_identity_key', 'matrix', 'in_vivo_in_vitro']):
        if len(g) < 2: continue
        records.append(dict(zip(columns[:3], key)) | {'n_records': len(g), 'n_publications': g.publication_id[g.publication_id.map(known)].nunique(),
                       'n_sources': g.source.nunique(), 'min_seconds': g.half_life_seconds.min(),
                       'max_seconds': g.half_life_seconds.max(), 'max_min_ratio': g.half_life_seconds.max()/g.half_life_seconds.min(),
                       'interpretation': 'Between-record spread; mixed protocols and shared-source representations may contribute. Not a replicate CV.'})
    return pd.DataFrame(records, columns=columns)


def feature_summary(frame):
    records = []
    for (source, view), g in frame[frame.is_human_serum].groupby(['source', 'dataset_view']):
        for feature in FEATURES:
            v = g[feature].dropna()
            records.append({'dataset': source+'/'+view, 'feature': feature, 'n': len(v),
                            'median': v.median(), 'q25': v.quantile(.25), 'q75': v.quantile(.75),
                            'min': v.min(), 'max': v.max(), 'descriptor_scope': 'reference unmodified backbone'})
    return pd.DataFrame(records)


def split_manifest(frame):
    """Give a conservative grouping unit for later baseline evaluation.

    Every identical sequence and every documented publication is connected.
    This prevents cross-source reuse and chemistry analogues of a sequence from
    leaking across folds. Sequence-similarity clustering is still needed before
    testing generalization to unseen families; no random-row benchmark is run.
    """
    records = frame[frame.is_human_serum & ~frame.is_redundant_representation].copy()
    parent = {x: x for x in records.observation_id}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(ids):
        ids = sorted(ids)
        for x in ids[1:]: parent[find(x)] = find(ids[0])
    for _, group in records.groupby('sequence'): union(group.observation_id.tolist())
    for _, group in records[records.publication_id.map(known)].groupby('publication_id'): union(group.observation_id.tolist())
    records['evaluation_group'] = records.observation_id.map(find)
    records['split_status'] = 'grouping_only; no folds assigned; similarity clustering pending'
    return records[['observation_id', 'source', 'sequence', 'publication_id', 'baseline_eligible', 'evaluation_group', 'split_status']]


def write_reports(frame, folder):
    folder.mkdir(parents=True, exist_ok=True)
    reports = {'coverage': coverage(frame), 'documentation': documentation(frame),
               'overlap_all_sources': overlaps(frame), 'overlap_human_serum': overlaps(frame, human_serum=True),
               'linked_label_comparison': linked_labels(frame), 'measurement_spread': measurement_spread(frame),
               'physicochemical_summary': feature_summary(frame), 'evaluation_groups': split_manifest(frame),
               'excluded_from_strict_baseline': frame[frame.exclusion_reasons.ne('')][['observation_id', 'source', 'dataset_view', 'species', 'matrix', 'relation', 'measurement_kind', 'exclusion_reasons']]}
    for name, report in reports.items(): report.to_csv(folder / f'{name}.tsv', sep='\t', index=False)
    return reports


def cavaco_comparison(frame, source):
    """Label audit on shared sequences, with chemistry and context explicitly checked.

    This is not an independent validation when an author dataset reused Cavaco.
    Missing CI in the target dataset is never filled from a sequence match.
    """
    reference = frame[frame.source.eq('cavaco')]
    target = frame[frame.source.eq(source)]
    merged = target.merge(reference, on='sequence', suffixes=('_source', '_cavaco'))
    columns = ['source', 'source_observation_id', 'cavaco_observation_id', 'sequence',
               'source_seconds', 'cavaco_seconds', 'ci95_lower_seconds', 'ci95_upper_seconds',
               'difference_seconds', 'relative_difference', 'inside_cavaco_ci95',
               'chemistry_match', 'same_environment', 'source_c_terminal', 'cavaco_c_terminal',
               'interpretation']
    records = []
    chemistry = ['chirality', 'topology', 'n_terminal_modification', 'c_terminal_modification', 'chemical_modifications']
    for _, r in merged.iterrows():
        point = r.half_life_seconds_source if r.relation_source == 'eq' else np.nan
        same_env = all(r[f+'_source'] == r[f+'_cavaco'] for f in ['species', 'matrix', 'in_vivo_in_vitro'])
        match = bool(r.chemical_identity_complete_source and r.chemical_identity_complete_cavaco and
                     all(r[f+'_source'] == r[f+'_cavaco'] for f in chemistry))
        records.append({'source': source, 'source_observation_id': r.observation_id_source,
                        'cavaco_observation_id': r.observation_id_cavaco, 'sequence': r.sequence,
                        'source_seconds': point, 'cavaco_seconds': r.half_life_seconds_cavaco,
                        'ci95_lower_seconds': r.ci_lower_seconds_cavaco, 'ci95_upper_seconds': r.ci_upper_seconds_cavaco,
                        'difference_seconds': point-r.half_life_seconds_cavaco,
                        'relative_difference': point/r.half_life_seconds_cavaco-1,
                        'inside_cavaco_ci95': bool(r.ci_lower_seconds_cavaco <= point <= r.ci_upper_seconds_cavaco) if pd.notna(point) else None,
                        'chemistry_match': match, 'same_environment': same_env,
                        'source_c_terminal': r.c_terminal_modification_source, 'cavaco_c_terminal': r.c_terminal_modification_cavaco,
                        'interpretation': 'Shared-sequence label audit; source reuse and chemistry differences preclude independent validation.'})
    return pd.DataFrame(records, columns=columns)


def dilution_pairs(frame):
    """Explore pairs within a DRAMP parent record; no model or labels rescaled."""
    records = []
    exact = frame[frame.source.eq('dramp') & frame.regression_label_available & frame.matrix_fraction.notna()]
    for parent, group in exact.groupby('source_record_id'):
        for (_, a), (_, b) in combinations(group.iterrows(), 2):
            if a.matrix_fraction == b.matrix_fraction or a.species != b.species or a.matrix != b.matrix:
                continue
            low, high = (a, b) if a.matrix_fraction < b.matrix_fraction else (b, a)
            records.append({'source_record_id': parent, 'low_fraction_observation': low.observation_id,
                            'high_fraction_observation': high.observation_id, 'low_fraction': low.matrix_fraction,
                            'high_fraction': high.matrix_fraction, 'low_fraction_half_life_seconds': low.half_life_seconds,
                            'high_fraction_half_life_seconds': high.half_life_seconds,
                            'observed_half_life_ratio': low.half_life_seconds/high.half_life_seconds,
                            'linear_dilution_expected_ratio': high.matrix_fraction/low.matrix_fraction,
                            'low_fraction_neat_proxy_seconds': low.matrix_fraction*low.half_life_seconds,
                            'high_fraction_neat_proxy_seconds': high.matrix_fraction*high.half_life_seconds,
                            'interpretation': 'Exploratory constant-hazard hypothesis; error kind unknown; no peptide-concentration adjustment.'})
    columns = ['source_record_id','low_fraction_observation','high_fraction_observation','low_fraction','high_fraction',
               'low_fraction_half_life_seconds','high_fraction_half_life_seconds','observed_half_life_ratio',
               'linear_dilution_expected_ratio','low_fraction_neat_proxy_seconds','high_fraction_neat_proxy_seconds','interpretation']
    return pd.DataFrame(records, columns=columns)
