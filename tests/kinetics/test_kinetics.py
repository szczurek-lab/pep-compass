"""Scientific identity, missingness and relational integrity across the pipeline."""
import copy
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets/tools/proteolysis_data_manager/src'))

from proteolysis_data_manager.acquisition.kinetics.brenda import import_snapshot, verify_snapshot
from proteolysis_data_manager.integration.kinetics.adapters.brenda import (
    adapt_records, context_windows, parse_value, scalar_condition,
)
from proteolysis_data_manager.integration.kinetics.normalization import normalize_brenda
from proteolysis_data_manager.integration.kinetics.reader import KineticsDataset, load_kinetics_data
from proteolysis_data_manager.integration.kinetics.schema import arrow_table
from proteolysis_data_manager.protocols.kinetics.filtering import evaluate_protocol, export_protocol
from proteolysis_data_manager.utils.io import sha256


@pytest.fixture
def record():
    return {
        'recommended_name': 'Test peptidase',
        'protein': {'1': {'organism': 'Homo sapiens', 'accessions': ['P00001']},
                    '2': {'organism': 'Mus musculus', 'accessions': ['P00002']}},
        'reference': {'10': {'pmid': 12345}, '11': {'title': 'Unindexed source'}},
        'turnover_number': [
            {'value': '2 {Ala-Gly}', 'proteins': ['1', '2'], 'references': ['10'], 'comment': 'pH 7.4, 37°C'},
            {'value': '2-8 {Ala-Gly}', 'proteins': ['2'], 'references': ['10'], 'comment': 'mutant T348Y, pH 7.4, 37°C'},
            {'value': '5 ± 1 {Ala-Gly}', 'proteins': ['1'], 'references': ['10', '11']},
            {'value': '-999 {Ala-Gly}', 'proteins': ['1']},
        ],
        'km_value': [{'value': '0.5 {Ala-Gly}', 'proteins': ['1'], 'references': ['10']}],
        'kcat_km_value': [{'value': '3 {Ala-Gly}', 'proteins': ['1'], 'references': ['10']}],
        # Generic optima must not populate assay fields.
        'ph_optimum': [{'value': '9.5', 'proteins': ['1']}],
        'temperature_optimum': [{'value': '60', 'proteins': ['1']}],
    }


@pytest.fixture
def dataset(record):
    rows, _ = adapt_records([('3.4.21.5', record)])
    return KineticsDataset({name: arrow_table(name, values).to_pandas() for name, values in rows.items()}).validate()


@pytest.fixture
def config():
    return {'condition_key': 'human_enzyme', 'enzyme_species': 'Homo sapiens',
            'experiment_filters': {'species': None, 'matrix': None, 'in_vivo_in_vitro': None},
            'acceptance_bounds': {'ph': None, 'temperature_c': None},
            'missing_condition_policy': 'retain', 'references': {'ph': 7.4, 'temperature_c': 37}}


def test_all_species_source_contexts_units_and_uncertainty(dataset):
    m = dataset['kinetic_measurements']
    assert len(m) == 6
    assert dataset['enzymes'].enzyme_species.eq('Mus musculus').any()
    assert m.experiment_id.nunique() == 6  # same PMID/name never proves shared setup
    assert m.loc[m.parameter_type.eq('km'), 'value_numeric'].iloc[0] == pytest.approx(.0005)
    assert m.loc[m.parameter_type.eq('kcat_km'), 'value_numeric'].iloc[0] == pytest.approx(3000)
    assert m.value_numeric.isna().sum() == 2  # range + sentinel survive
    assert m[['sd', 'ci_lower', 'ci_upper', 'ci_level', 'replicate_count']].isna().all().all()
    assert dataset['substrates'][['sequence', 'chirality', 'topology']].isna().all().all()
    assert dataset['experiments'][['species', 'matrix', 'in_vivo_in_vitro']].isna().all().all()
    assert dataset['experiments'].publication_urls.dropna().map(lambda v: 'https://pubmed.ncbi.nlm.nih.gov/12345/' in v).all()
    assert dataset['experiments'].temperature_c.dropna().tolist() == [37, 37]
    assert dataset['cleavage_sites'].empty


@pytest.mark.parametrize('raw,relation', [('2-8', 'range'), ('<3', 'lt'), ('>1', 'gt'), ('about 5', 'approx'), ('-999','unknown'), ('1e309','unknown')])
def test_non_points_never_become_midpoints(raw, relation):
    assert parse_value(raw) == (None, relation)


@pytest.mark.parametrize('text,field', [('pH 6-8','ph'), ('pH 7.0 and 8.0','ph'), ('pH 7, pH 8','ph'),
                                     ('optimal pH 7.4','ph'), ('30-40°C','temperature_c'),
                                     ('30 to 40°C','temperature_c'), ('storage at 4°C','temperature_c')])
def test_ambiguous_or_non_assay_conditions_stay_null(text, field):
    assert scalar_condition(text, field) is None


def test_group_links_do_not_multiply_values_or_become_human_serum(dataset, config):
    view = dataset.measurements_view()
    assert len(view) == 6 and view.measurement_id.is_unique
    assert any(v == ['Homo sapiens', 'Mus musculus'] for v in view.enzyme_species)
    selected = evaluate_protocol(dataset, config)
    assert selected.accepted.sum() == 4
    assert selected.numeric_model_eligible.sum() == 3
    assert selected.loc[selected.accepted, 'value_numeric'].isna().sum() == 1
    serum_config = copy.deepcopy(config)
    serum_config['experiment_filters'] = {'species': 'human', 'matrix': 'serum'}
    assert not evaluate_protocol(dataset, serum_config).accepted.any()
    gated = copy.deepcopy(config)
    gated['acceptance_bounds']['ph'] = [7.3, 7.5]
    assert evaluate_protocol(dataset, gated).accepted.sum() == 4
    gated['missing_condition_policy'] = 'reject'
    assert not evaluate_protocol(dataset, gated).accepted.any()


def test_reader_rejects_foreign_key_and_group_membership_errors(dataset):
    broken = copy.deepcopy(dataset)
    broken['kinetic_measurements'].loc[0, 'substrate_id'] = 'missing'
    with pytest.raises(ValueError, match='foreign key'):
        broken.validate()
    broken = copy.deepcopy(dataset)
    broken.tables['experiment_substrates_relation'] = broken['experiment_substrates_relation'].iloc[1:].copy()
    with pytest.raises(ValueError, match='Substrate outside experiment'):
        broken.validate()


def test_multiple_sites_do_not_clone_whole_substrate_parameter(dataset):
    measurement = dataset['kinetic_measurements'].iloc[0]
    enzyme = dataset['experiment_enzymes_relation'].iloc[0].enzyme_id
    sites = []
    for index in [4, 5]:
        left, right, whole = context_windows('ACDEFGHIK', index)
        sites.append({'site_id': f'site-{index}', 'experiment_id': measurement.experiment_id,
                      'substrate_id': measurement.substrate_id, 'enzyme_id': enzyme,
                      'position_reported': index, 'coordinate_basis': 'substrate_sequence',
                      'cleavage_context_p4_to_p1': left, 'cleavage_context_p1prime_to_p4prime': right,
                      'cleavage_context_8aa': whole, 'context_origin': 'derived_from_reported_position'})
    dataset.tables['cleavage_sites'] = arrow_table('cleavage_sites', sites).to_pandas()
    dataset.validate()
    assert len(dataset.measurements_view()) == 6
    dataset.tables['measurement_cleavage_sites_relation'] = arrow_table('measurement_cleavage_sites_relation', [
        {'measurement_id': measurement.measurement_id, 'site_id': 'site-4'},
        {'measurement_id': measurement.measurement_id, 'site_id': 'site-5'},
    ]).to_pandas()
    with pytest.raises(ValueError, match='cannot be assigned'):
        dataset.validate()
    dataset['kinetic_measurements'].loc[0, 'measurement_scope'] = 'site_set'
    dataset.validate()
    view = dataset.measurements_view()
    assert len(view) == 6 and view.site_ids.iloc[0] == ['site-4', 'site-5']
    dataset['kinetic_measurements'].loc[0, 'experiment_id'] = dataset['experiments'].iloc[1].experiment_id
    with pytest.raises(ValueError, match='context mismatch'):
        dataset.validate()


def test_subsetting_keeps_closed_relations(dataset):
    target = dataset['enzymes'].loc[dataset['enzymes'].enzyme_species.eq('Mus musculus'), 'enzyme_id'].iloc[0]
    selected = dataset.for_enzyme(target)
    assert len(selected['kinetic_measurements']) == 1
    selected.validate()
    empty = dataset.subset_experiments([])
    assert all(table.empty for table in empty.tables.values())
    empty.validate()


def test_overview_accounts_for_entries_without_protein_links(record):
    from proteolysis_data_manager.analysis.kinetics.overview import species_coverage
    record['km_value'].append({'value': '0.2 {Gly}', 'proteins': []})
    rows, _ = adapt_records([('3.4.21.5', record)])
    dataset = KineticsDataset({name: arrow_table(name, values).to_pandas() for name, values in rows.items()}).validate()
    coverage = species_coverage(dataset)
    assert coverage.loc[coverage.enzyme_species.eq('no_source_protein_link'), 'measurements'].iloc[0] == 1
    assert len(dataset['kinetic_measurements']) == 7


def test_import_normalize_roundtrip_and_protocol(tmp_path, record, config):
    archive = tmp_path / 'brenda_test.tar.gz'
    payload = json.dumps({'data': {'1.1.1.1': record, '3.4.21.5': record}}).encode()
    with tarfile.open(archive, 'w:gz') as tar:
        member = tarfile.TarInfo('brenda.json'); member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({'files': [{'sha256': sha256(archive)}], 'retrieved_at': '2020-01-01', 'source_version_or_accession': 'test'}))
    root = tmp_path / 'data'
    first = import_snapshot(archive, root, receipt=receipt)
    assert first == import_snapshot(archive, root, receipt=receipt)
    assert first['retrieved_at'] == '2020-01-01' and first['status'] == 'imported_local'
    dataset, manifest = normalize_brenda(root)
    assert manifest['raw_kinetic_entries'] == 6
    loaded = load_kinetics_data(root)
    assert len(loaded['kinetic_measurements']) == 6
    rows = evaluate_protocol(loaded, config)
    selected, _ = export_protocol(loaded, rows, root, config)
    assert len(selected['kinetic_measurements']) == 4
    reviewed = load_kinetics_data(root, condition='human_enzyme')
    assert len(reviewed['kinetic_measurements']) == 4
    path = root / 'normalized/enzymes.parquet'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Changed kinetics table'):
        load_kinetics_data(root)
    raw = root / first['raw_path']
    raw.write_bytes(raw.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed after acquisition'):
        verify_snapshot(root)
