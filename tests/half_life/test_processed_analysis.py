"""Check reported-context overlap and figure data without sequence-only pooling."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'assets/tools/proteolysis_data_manager/src'))
from proteolysis_data_manager.integration.half_life.adapters.common import row
from proteolysis_data_manager.integration.half_life.adapters.schema import adapt_records
from proteolysis_data_manager.analysis.half_life.processed import summarize_processed


def sample(source, identifier, *, sequence='AAA', temperature=37, c_terminal='free', value='30'):
    return row(source, 'original', identifier, {}, sequence, value, 'minutes',
               species='human', matrix='serum', in_vivo_in_vitro='in vitro',
               chirality='l', topology='linear', n_terminal_modification='free',
               c_terminal_modification=c_terminal, chemical_modifications='free',
               temperature_c=temperature, matrix_fraction=.5)


def test_overlap_counts_text_separately_from_chemistry():
    dataset = adapt_records([sample('a', '1'), sample('b', '2', c_terminal='amidation')])
    tables = summarize_processed(dataset.sequences_half_time, dataset.half_time_experiments)
    assert tables['sequence_overlap'].query("source_a == 'a' and source_b == 'b'").shared_count.item() == 1
    assert tables['peptide_overlap'].query("source_a == 'a' and source_b == 'b'").shared_count.item() == 0
    assert tables['shared_measurements'].empty


def test_shared_conditions_exclude_known_mismatches_and_keep_missing_flags():
    dataset = adapt_records([sample('a', '1'), sample('b', '2', value='60'),
                            sample('c', '3', temperature=40), sample('d', '4', value='<30')])
    tables = summarize_processed(dataset.sequences_half_time, dataset.half_time_experiments)
    shared = tables['shared_measurements']
    assert set(shared.source) == {'a', 'b'}
    assert not shared.condition_metadata_complete.any()  # pH/concentration were not reported.
    assert tables['shared_comparisons'].max_to_min_ratio.item() == 2
    assert set(tables['duration_values'].source) == {'a', 'b', 'c'}
    assert set(tables['duration_values'].stability_class) == {'medium'}


def test_unknown_core_context_does_not_establish_cross_source_agreement():
    first, second = sample('a', '1'), sample('b', '2')
    first['in_vivo_in_vitro'] = second['in_vivo_in_vitro'] = 'unknown'
    dataset = adapt_records([first, second])
    tables = summarize_processed(dataset.sequences_half_time, dataset.half_time_experiments)
    assert tables['shared_measurements'].empty
    assert tables['conditions'].observations.sum() == 2
    assert tables['conditions'].in_vivo_in_vitro.isna().all()


def test_missingness_denominator_is_source_observations_and_overlap_is_symmetric():
    dataset = adapt_records([sample('a', '1'), sample('a', '2', temperature=None), sample('b', '3')])
    tables = summarize_processed(dataset.sequences_half_time, dataset.half_time_experiments)
    missing = tables['missingness'].query("source == 'a' and field == 'temperature_c'").iloc[0]
    assert (missing.observations, missing.missing, missing.missing_percent) == (2, 1, 50)
    matrix = tables['sequence_overlap'].pivot(index='source_a', columns='source_b', values='shared_count')
    assert (matrix.to_numpy() == matrix.to_numpy().T).all()
    assert tables['source_counts'].observations.sum() == 3
    assert tables['duration_counts'].observations.sum() == 3
