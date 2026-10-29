"""Auditable summaries of the linked peptide and experiment datasets."""
from __future__ import annotations

from itertools import combinations_with_replacement

import pandas as pd

from proteolysis_data_manager.integration.half_life.adapters.schema import HalfTimeDataset
from proteolysis_data_manager.integration.half_life.labels import point_class
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)
CONDITION_COLUMNS = ['species', 'matrix', 'in_vivo_in_vitro', 'matrix_fraction',
                     'temperature_c', 'ph', 'peptide_concentration_um']
CORE_CONDITIONS = CONDITION_COLUMNS[:3]
METADATA_COLUMNS = [
    'publication_id', 'assay', 'species', 'matrix', 'in_vivo_in_vitro',
    'matrix_fraction', 'temperature_c', 'ph', 'peptide_concentration_um',
    'half_life_seconds', 'ci_lower_seconds', 'ci_upper_seconds', 'ci_level',
    'uncertainty_value_seconds', 'replicate_count', 'chirality', 'topology',
    'n_terminal_modification', 'c_terminal_modification', 'other_modifications',
]


def context_label(record, *, detailed=False):
    """Format reported conditions, showing missing values explicitly.

    :param mapping record: A row containing condition fields.
    :param bool detailed: Include reported numeric assay conditions.
    :return: Human-readable context without substituted defaults.
    :rtype: str
    """
    def value(field, suffix=''):
        v = record[field]
        return 'not reported' if pd.isna(v) else f'{v:g}{suffix}' if isinstance(v, (float, int)) else str(v)
    label = ' | '.join(value(c) for c in CORE_CONDITIONS)
    if detailed:
        label += '\n' + '; '.join(f'{name}={value(field, suffix)}' for field, name, suffix in [
            ('matrix_fraction', 'fraction', ''), ('temperature_c', 'T', ' °C'),
            ('ph', 'pH', ''), ('peptide_concentration_um', 'peptide', ' µM'),
        ])
    return label


def _overlap(frame, key):
    groups = {name: set(rows[key].dropna()) for name, rows in frame.groupby('source')}
    rows = []
    for first, second in combinations_with_replacement(sorted(groups), 2):
        count = len(groups[first] & groups[second])
        rows.append({'source_a': first, 'source_b': second, 'shared_count': count})
        if first != second:
            rows.append({'source_a': second, 'source_b': first, 'shared_count': count})
    return pd.DataFrame(rows, columns=['source_a', 'source_b', 'shared_count'])


def shared_measurements(frame, *, species=('human', 'mouse')):
    """Select cross-source points with equal peptide and reported conditions.

    :param pandas.DataFrame frame: Experiments joined to the peptide registry.
    :param tuple species: Species to compare.
    :return: All matching observations, including missing numeric metadata flags.
    :rtype: pandas.DataFrame

    Only exact positive half-lives and documented species/matrix/mode enter.
    All seven condition fields and the peptide identity form the grouping key.
    Matching missing numeric fields means equal reported context, not proven
    experimental equivalence. Complexity is O(N log N) time and O(N) space for
    N observations; sorting provides deterministic comparison IDs.
    """
    keep = (frame.species.isin(species) & frame[CORE_CONDITIONS].notna().all(axis=1)
            & frame.measurement_kind.eq('half_life') & frame.relation.eq('eq')
            & frame.half_life_seconds.gt(0))
    selected = frame.loc[keep].copy()
    key = ['sequence_id', *CONDITION_COLUMNS]
    n_sources = selected.groupby(key, dropna=False).source.transform('nunique')
    selected = selected.loc[n_sources.ge(2)].copy()
    selected['condition_metadata_complete'] = selected[CONDITION_COLUMNS].notna().all(axis=1)
    if selected.empty:
        selected['comparison_id'] = pd.Series(dtype='string')
        selected['context_label'] = pd.Series(dtype='string')
        return selected
    selected = selected.sort_values([*key, 'source', 'experiment_id'], na_position='last')
    group_ids = selected.groupby(key, dropna=False, sort=True).ngroup()
    selected['comparison_id'] = group_ids.map(lambda i: f'C{i + 1:03d}')
    selected['context_label'] = selected.apply(lambda r: context_label(r, detailed=True), axis=1)
    return selected.reset_index(drop=True)


def summarize_processed(sequences, experiments):
    """Compute the canonical tables underlying the five dataset analyses.

    :param pandas.DataFrame sequences: The seven-column peptide registry.
    :param pandas.DataFrame experiments: The experiment table.
    :return: Named tables for coverage, overlaps, conditions and exact durations.
    :rtype: dict[str, pandas.DataFrame]
    :raises ValueError: If schemas or the many-to-one peptide relationship fail.
    """
    HalfTimeDataset(sequences, experiments).validate()
    frame = experiments.merge(sequences, on='sequence_id', how='left', validate='many_to_one')
    logger.info('Summarizing %s observations from %s sources', len(frame), frame.source.nunique())
    source_rows, missing_rows = [], []
    total_observations = len(frame)
    total_peptide_descriptions = frame.sequence_id.nunique()
    total_sequence_strings = frame.sequence.nunique()
    for source, rows in frame.groupby('source', sort=True):
        observations = len(rows)
        source_records = rows.source_record_id.nunique()
        reported_peptides = rows.sequence_id.nunique()
        sequence_strings = rows.sequence.nunique()
        source_rows.append({
            'source': source,
            'observations': observations,
            'observations_share_percent': 100 * observations / total_observations,
            'source_records': source_records,
            'source_records_within_source_percent': 100 * source_records / observations,
            'reported_peptides': reported_peptides,
            'reported_peptides_union_coverage_percent':
                100 * reported_peptides / total_peptide_descriptions,
            'sequence_strings': sequence_strings,
            'sequence_strings_union_coverage_percent':
                100 * sequence_strings / total_sequence_strings,
        })
        for column in METADATA_COLUMNS:
            missing = int(rows[column].isna().sum())
            missing_rows.append({'source': source, 'field': column, 'observations': len(rows),
                                 'missing': missing, 'missing_percent': 100 * missing / len(rows)})
    conditions = frame.groupby(CORE_CONDITIONS, dropna=False).agg(
        observations=('experiment_id', 'size'), reported_peptides=('sequence_id', 'nunique'),
        sources=('source', 'nunique'),
    ).reset_index().sort_values('observations', ascending=False)
    conditions['condition_label'] = conditions.apply(context_label, axis=1)
    shared = shared_measurements(frame)
    if shared.empty:
        shared_summary = pd.DataFrame(columns=['comparison_id', 'sequence', 'context_label', 'sources',
                                              'observations', 'min_seconds', 'max_seconds', 'max_to_min_ratio'])
    else:
        shared_summary = shared.groupby(['comparison_id', 'sequence', 'context_label'], sort=True).agg(
            sources=('source', 'nunique'), observations=('experiment_id', 'size'),
            min_seconds=('half_life_seconds', 'min'), max_seconds=('half_life_seconds', 'max'),
        ).reset_index()
        shared_summary['max_to_min_ratio'] = shared_summary.max_seconds / shared_summary.min_seconds
    exact = frame.loc[frame.measurement_kind.eq('half_life') & frame.relation.eq('eq')
                      & frame.half_life_seconds.gt(0)].copy()
    exact['stability_class'] = exact.half_life_seconds.map(point_class)
    exact['condition_label'] = exact.apply(context_label, axis=1)
    class_counts = exact.groupby(['source', 'condition_label', 'stability_class'], observed=True).size().rename('observations').reset_index()
    return {
        'source_counts': pd.DataFrame(source_rows), 'missingness': pd.DataFrame(missing_rows),
        'sequence_overlap': _overlap(frame, 'sequence'), 'peptide_overlap': _overlap(frame, 'sequence_id'),
        'conditions': conditions.reset_index(drop=True), 'shared_measurements': shared,
        'shared_comparisons': shared_summary, 'duration_values': exact,
        'duration_counts': class_counts,
    }
