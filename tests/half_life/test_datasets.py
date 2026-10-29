"""Contract, identity and measurement checks for the two half-time datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from openpyxl import Workbook
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'assets/tools/proteolysis_data_manager/src'))

from proteolysis_data_manager.integration.half_life.adapters import common, dramp
from proteolysis_data_manager.integration.half_life.adapters.schema import (
    HalfTimeDataset, SEQUENCE_COLUMNS, EXPERIMENT_COLUMNS, adapt_records,
)
from proteolysis_data_manager.integration.half_life.datasets import (
    build_half_time_datasets, read_half_time_datasets,
)


@pytest.fixture
def observation():
    """Construct small explicit source records without invoking descriptors."""
    def make(identifier='1', value='2', unit='minutes', source='example', sequence='AAA', **metadata):
        chemistry = dict(chirality='l', topology='linear', n_terminal_modification='free',
                         c_terminal_modification='free', chemical_modifications='free')
        return common.row(source, 'original', identifier, {}, sequence, value, unit,
                          include_features=False, **(chemistry | metadata))
    return make


def test_split_contract_links_observations_without_generated_features(observation):
    dataset = adapt_records([observation('1'), observation('2', temperature_c=40)])
    assert list(dataset.sequences_half_time) == SEQUENCE_COLUMNS
    assert list(dataset.half_time_experiments) == EXPERIMENT_COLUMNS
    assert len(dataset.sequences_half_time) == 1
    assert len(dataset.half_time_experiments) == 2
    assert set(dataset.half_time_experiments.sequence_id) == set(dataset.sequences_half_time.sequence_id)
    assert dataset.half_time_experiments.half_life_seconds.eq(120).all()
    assert dataset.half_time_experiments.temperature_c.isna().sum() == 1


@pytest.mark.parametrize('change', [
    {'sequence': 'aAA'}, {'chirality': 'd'}, {'topology': 'cyclic'},
    {'n_terminal_modification': 'acetylation'}, {'c_terminal_modification': 'amidation'},
    {'chemical_modifications': 'lipidation'}, {'chirality': 'unknown'},
])
def test_sequence_identity_preserves_case_and_every_chemistry_field(observation, change):
    dataset = adapt_records([observation('1'), observation('2', **change)])
    assert len(dataset.sequences_half_time) == 2
    assert dataset.half_time_experiments.sequence_id.nunique() == 2


def test_identical_descriptions_deduplicate_but_sources_and_conditions_survive(observation):
    a = observation('1', source='a', temperature_c=37)
    b = observation('1', source='b', temperature_c=40)
    forward, backward = adapt_records([a, b]), adapt_records([b, a])
    pd.testing.assert_frame_equal(forward.sequences_half_time, backward.sequences_half_time)
    pd.testing.assert_frame_equal(forward.half_time_experiments, backward.half_time_experiments)
    assert len(forward.sequences_half_time) == 1
    assert forward.half_time_experiments.temperature_c.tolist() == [37, 40]
    assert forward.half_time_experiments.source.tolist() == ['a', 'b']


def test_keys_and_output_are_stable_when_input_order_changes(observation):
    records = [observation('1'), observation('2', sequence='CCC'), observation('3')]
    a, b = adapt_records(records), adapt_records(reversed(records))
    pd.testing.assert_frame_equal(a.sequences_half_time, b.sequences_half_time)
    pd.testing.assert_frame_equal(a.half_time_experiments, b.half_time_experiments)


@pytest.mark.parametrize('value,relation,point,lower,upper', [
    ('2', 'eq', 120, None, None), ('~2', 'approx', 120, None, None),
    ('<2', 'lt', None, None, 120), ('>=2', 'ge', None, 120, None),
    ('2–4', 'interval', None, 120, 240), ('N.A.', 'unknown', None, None, None),
])
def test_readouts_preserve_relation_units_and_bounds(observation, value, relation, point, lower, upper):
    row = adapt_records([observation(value=value)]).half_time_experiments.iloc[0]
    assert row.relation == relation
    for field, expected in [('half_life_seconds', point), ('lower_bound_seconds', lower), ('upper_bound_seconds', upper)]:
        assert pd.isna(row[field]) if expected is None else row[field] == expected


def test_ci_is_not_used_as_an_invented_point_or_sd(observation):
    record = observation(value='N.A.', ci_lower_seconds=100, ci_upper_seconds=200,
                         ci_confidence_level=.95, uncertainty_kind='reported_ci', n_replicates=3)
    row = adapt_records([record]).half_time_experiments.iloc[0]
    assert pd.isna(row.half_life_seconds)
    assert (row.ci_lower_seconds, row.ci_upper_seconds, row.ci_level, row.replicate_count) == (100, 200, .95, 3)
    assert row.uncertainty_kind == 'reported_ci'
    assert pd.isna(row.uncertainty_value_seconds)


def test_unspecified_error_is_not_renamed_sd(observation):
    row = adapt_records([observation(value='2 ± 0.5')]).half_time_experiments.iloc[0]
    assert row.half_life_seconds == 120
    assert row.uncertainty_kind == 'unspecified_plus_minus'
    assert row.uncertainty_value_seconds == 30
    assert pd.isna(row.ci_lower_seconds)


def test_condition_specific_dramp_clauses_remain_two_experiments():
    records = dramp.dramp_rows({
        'DRAMP_ID': 'DRAMP29881', 'Sequence': 'AAA', 'Type': 'Mouse serum',
        'half_life': 'The half-live is 68 ±15 min in 25% aqueous mouse serum, 25±4 min in full mouse serum.',
    })
    dataset = adapt_records(records)
    assert len(dataset.sequences_half_time) == 1
    experiments = dataset.half_time_experiments
    assert experiments.experiment_id.nunique() == 2
    assert experiments.source_record_id.eq('DRAMP29881').all()
    assert experiments.matrix_fraction.tolist() == [.25, 1]
    assert experiments.half_life_seconds.tolist() == [4080, 1500]
    assert experiments.uncertainty_value_seconds.tolist() == [900, 240]


def test_non_half_life_readout_stays_unlabelled(observation):
    record = observation()
    record.update(common.measurement('70% remaining after 6 hours', prose=True))
    row = adapt_records([record]).half_time_experiments.iloc[0]
    assert row.measurement_kind == 'remaining_fraction'
    assert pd.isna(row.half_life_seconds)


def test_validation_rejects_foreign_and_duplicate_keys(observation):
    data = adapt_records([observation()])
    invalid = HalfTimeDataset(data.sequences_half_time.iloc[:0], data.half_time_experiments)
    with pytest.raises(ValueError, match='no peptide record'):
        invalid.validate()
    with pytest.raises(ValueError, match='duplicate experiment_id'):
        adapt_records([observation(), observation()])


def test_empty_union_retains_typed_contract():
    dataset = adapt_records([])
    assert dataset.sequences_half_time.empty and dataset.half_time_experiments.empty
    assert list(dataset.sequences_half_time) == SEQUENCE_COLUMNS
    assert str(dataset.half_time_experiments.half_life_seconds.dtype) == 'Float64'


@pytest.fixture
def acquired_sources(tmp_path):
    """Small source-format fixtures with real checksums and Cavaco provenance.

    Worksheets preserve the published headers used by each adapter. The Cavaco
    fixture represents printed half-life/CI labels linked to XML Table 1.
    """
    from proteolysis_data_manager.acquisition.half_life import SOURCES

    root, raw = tmp_path / 'series', tmp_path / 'series/raw'
    for source in SOURCES:
        (raw / source).mkdir(parents=True)

    base = dict(seq='AAA', half_life='2', units_half='minutes', chiral='L', lin_cyc='Linear',
                nter='Free', cter='Free', chem_mod='None', name='Example', pmid='123',
                test_sample='Human serum', vivo_vitro='in vitro', assay='HPLC')
    for partition, identifier in [('natural', '1'), ('modified', '2')]:
        (raw / f'peplife2/{partition}.json').write_text(json.dumps({
            'status': 200, 'count': 1, 'data': [base | {'id': identifier}],
        }))
    (raw / 'peplife2/complete.fasta').write_text('>1\nAAA\n>2\nAAA\n')

    def workbook(path, record):
        wb = Workbook()
        wb.active.append(list(record))
        wb.active.append(list(record.values()))
        wb.save(path)

    workbook(raw / 'pepmsnd/dataset.xlsx', {'ID': 'p1', 'Sequence': 'AAA', 'T1/2(hour)': 1,
             'Test Sample': 'Human serum', 'In-Vivo/Vitro': 'in vitro', 'SMILES': 'source-only', 'MolWt': 123})
    for view, column in [('natural', 'hl in seconds'), ('mixed', 'Half-life in seconds')]:
        workbook(raw / f'plifepred/{view}.xlsx', base | {'id.pdb': '1.pdb', column: 120})
    workbook(raw / 'dramp/stability_amps.xlsx', {'DRAMP_ID': 'd1', 'Sequence': 'AAA',
             'half_life': 'Half-life is 30 minutes in human serum.', 'Type': 'Human serum'})

    (raw / 'cavaco/article.xml').write_text('<article><table-wrap><label>Table 1</label>'
        '<table><tbody><tr><td>1</td><td>AAA</td></tr></tbody></table></table-wrap></article>')
    for name in ['article.pdf', 'supplement.pdf', 'supplement.docx']:
        (raw / 'cavaco' / name).write_bytes(b'%PDF-1.0 deterministic fixture')
    config = tmp_path / 'reviewed'
    config.mkdir()
    (config / 'cavaco_validation.tsv').write_text('peptide_id\tsequence\thalf_life_minutes\tci_lower_minutes\tci_upper_minutes\n1\tAAA\t48\t28\t78\n')
    (config / 'cavaco_validation.provenance.json').write_text(json.dumps({
        'article_pdf_sha256': hashlib.sha256((raw / 'cavaco/article.pdf').read_bytes()).hexdigest(),
    }))
    manifest = {'version': 1, 'sources': {source: {
        'source': source, 'version': 'fixture',
        'files': {name: {'sha256': hashlib.sha256((raw / source / name).read_bytes()).hexdigest()} for name in files},
    } for source, files in SOURCES.items()}}
    (raw / 'manifest.json').write_text(json.dumps(manifest))
    return root, config


def test_all_adapters_build_two_outputs_and_roundtrip_without_features(acquired_sources, monkeypatch):
    root, config = acquired_sources

    def reject_features(sequence):
        raise AssertionError('Dataset construction must not generate descriptors')

    monkeypatch.setattr(common, 'features', reject_features)
    before = {p: p.read_bytes() for p in (root / 'raw').rglob('*') if p.is_file()}
    dataset = build_half_time_datasets(root, config_dir=config)
    output = root / 'processed'
    assert sorted(p.name for p in output.iterdir()) == ['half_time_experiments.tsv', 'sequences_half_time.tsv']
    restored = read_half_time_datasets(output)
    assert len(restored.half_time_experiments) == 7
    assert set(restored.half_time_experiments.source) == {'peplife2', 'pepmsnd', 'plifepred', 'dramp', 'cavaco'}
    for name in ['sequences_half_time', 'half_time_experiments']:
        pd.testing.assert_frame_equal(getattr(dataset, name), getattr(restored, name), check_dtype=False)
    cavaco = restored.half_time_experiments.loc[restored.half_time_experiments.source.eq('cavaco')].iloc[0]
    assert (cavaco.half_life_seconds, cavaco.ci_lower_seconds, cavaco.ci_upper_seconds) == (2880, 1680, 4680)
    assert cavaco.peptide_concentration_um == 500
    assert all(p.read_bytes() == content for p, content in before.items())
    # Explicit output overrides must not write to the series' default folder.
    custom = root / 'custom'
    build_half_time_datasets(root, output_dir=custom, config_dir=config)
    assert len(read_half_time_datasets(custom).half_time_experiments) == 7


def test_tsv_reader_preserves_literal_na_sequence_and_zero_padded_ids(observation, tmp_path):
    dataset = adapt_records([observation('0001', sequence='NA')])
    for name in ['sequences_half_time', 'half_time_experiments']:
        getattr(dataset, name).to_csv(tmp_path / f'{name}.tsv', sep='\t', index=False)
    restored = read_half_time_datasets(tmp_path)
    assert restored.sequences_half_time.sequence.item() == 'NA'
    assert restored.half_time_experiments.source_record_id.item() == '0001'
