"""Byte-preserving MEROPS imports and lossless dictionary/matrix CSV joins."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'assets/tools/proteolysis_data_manager/src'))
from proteolysis_data_manager.acquisition.helpers import raw
from proteolysis_data_manager.acquisition.window_neighbourhood import merops
from proteolysis_data_manager.integration.window_neighbourhood.merops import COLUMNS, merge_merops


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """Create two indexed datasets with repeated codes and shuffled metadata rows."""
    source = tmp_path / 'originals'
    source.mkdir()
    positions = ['P4', 'P3', 'P2', 'P1', 'P1p', 'P2p', 'P3p', 'P4p']
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    entries = {}
    for dataset_id in (12, 3):
        folder = source / f'{dataset_id:02d}_source'
        folder.mkdir()
        entry = dict(dataset_index=dataset_id, folder=folder.name, scientific='Original species',
                     role='host', apex_index=None, label='Original dataset', taxids=['sp000001'], n_peptidases=2)
        entries[str(dataset_id)] = entry
        counts = (np.arange(320).reshape(2, 8, 20) + dataset_id).astype(np.float32)  # (N=2, P=8, A=20)
        logprob = -(np.arange(320, dtype=np.float32).reshape(2, 8, 20) + 1) / 10  # (N=2, P=8, A=20)
        enzymes = [{'row': row, 'code': code, 'name': 'Original "quoted", enzyme',
                    'n_cleavages': 100, 'serum_category': 'original category'}
                   for row, code in enumerate(('A01.001', 'S01.001'))]
        metadata = dict(positions=positions, amino_acids=amino_acids, peptidases=list(reversed(enzymes)))
        (folder / 'codes.json').write_text(json.dumps(metadata, indent=2) + '\r\n')
        np.save(folder / 'matrices_counts.npy', counts)
        np.save(folder / 'matrices_logprob.npy', logprob)
        (folder / 'matrices.pt').write_bytes(b'Original opaque PT bytes\r\n')
        for enzyme in enzymes:
            with (folder / f'{enzyme["code"]}.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['position', *amino_acids])
                writer.writerows([position, *counts[enzyme['row'], i]] for i, position in enumerate(positions))
    index = dict(n_datasets=2, by_index=entries, by_folder={entry['folder']: entry for entry in entries.values()},
                 positions=positions, amino_acids=amino_acids)
    (source / 'index.json').write_text(json.dumps(index, indent=2) + '\r\n')
    (source / 'README.md').write_bytes(b'Original bytes, including CRLF\r\n\xff')
    monkeypatch.setattr(raw, 'urlopen', lambda *args, **kwargs: pytest.fail('MEROPS import used the network'))
    return source, tmp_path / 'dataset'


def source_bytes(source):
    return {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob('*') if path.is_file()}


def test_original_dictionaries_and_every_nested_file_are_preserved(snapshot):
    source, root = snapshot
    original = source_bytes(source)
    receipt = merops.acquire(root / 'raw/merops', source_root=source)
    manifest = (root / 'raw/manifest.json').read_bytes()
    assert source_bytes(source) == original == source_bytes(root / 'raw/merops')
    assert merops.acquire(root / 'raw/merops', source_root=source) == receipt
    assert (root / 'raw/manifest.json').read_bytes() == manifest
    assert len(list(root.rglob('*manifest*.json'))) == 1
    for entry in receipt['files'].values():
        assert entry['status'] == 'imported_local'
        assert entry['retrieved_at'] is None
        assert entry['imported_at']
        assert entry['requested_url'].startswith('file:')


def test_invalid_refresh_keeps_the_previous_snapshot_and_receipt(snapshot):
    source, root = snapshot
    merops.acquire(root / 'raw/merops', source_root=source)
    before = source_bytes(root / 'raw')
    matrix = np.load(source / '03_source/matrices_counts.npy')
    matrix[0, 0, 0] += 1
    np.save(source / '03_source/matrices_counts.npy', matrix)
    with pytest.raises(ValueError, match='CSV and NPY'):
        merops.acquire(root / 'raw/merops', source_root=source, force=True)
    assert source_bytes(root / 'raw') == before


def test_registered_snapshot_is_reused_without_the_original_import_directory(snapshot, monkeypatch):
    source, root = snapshot
    receipt = merops.acquire(root / 'raw/merops', source_root=source)
    before = (root / 'raw/manifest.json').read_bytes()
    monkeypatch.setattr(merops, 'default_source_root', lambda: source / 'missing')
    assert merops.acquire(root / 'raw/merops') == receipt
    assert (root / 'raw/manifest.json').read_bytes() == before


@pytest.mark.parametrize('defect', ['rows', 'axes', 'shape', 'folders'])
def test_source_dictionary_inconsistencies_are_rejected(snapshot, defect):
    source, root = snapshot
    path = source / ('index.json' if defect == 'folders' else '03_source/codes.json')
    metadata = json.loads(path.read_text())
    if defect == 'rows':
        metadata['peptidases'][0]['row'] = metadata['peptidases'][1]['row']
    elif defect == 'axes':
        metadata['positions'].reverse()
    elif defect == 'folders':
        metadata['by_folder']['03_source']['dataset_index'] = 999
    else:
        np.save(source / '03_source/matrices_counts.npy', np.zeros((2, 20, 8)))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        merops.acquire(root / 'raw/merops', source_root=source)
    assert not (root / 'raw/manifest.json').exists()


def test_csv_preserves_all_cells_indices_and_repeated_codes(snapshot):
    source, root = snapshot
    originals = source_bytes(source)
    merops.acquire(root / 'raw/merops', source_root=source)
    before = source_bytes(root / 'raw')
    path = merge_merops(root)
    frame = pd.read_csv(path, float_precision='round_trip', keep_default_na=False)
    assert tuple(frame.columns) == COLUMNS
    assert len(frame) == 2 * 2 * 8 * 20
    assert not frame.duplicated(['dataset_index', 'peptidase_row', 'position_index', 'amino_acid_index']).any()
    assert frame['dataset_index'].drop_duplicates().tolist() == [3, 12]
    assert frame.groupby('code')['dataset_index'].nunique().to_dict() == {'A01.001': 2, 'S01.001': 2}
    for dataset_id, group in frame.groupby('dataset_index', sort=False):
        directory = source / f'{dataset_id:02d}_source'
        for column, kind in [('count', 'counts'), ('logprob', 'logprob')]:
            expected = np.load(directory / f'matrices_{kind}.npy')  # (N, P, A)
            actual = group[column].to_numpy(dtype=expected.dtype).reshape(expected.shape)  # (N, P, A)
            assert np.array_equal(actual, expected)
        assert group.groupby('peptidase_row')['code'].first().to_dict() == {0: 'A01.001', 1: 'S01.001'}
        assert group['name'].unique().tolist() == ['Original "quoted", enzyme']
        assert group['apex_index'].unique().tolist() == ['']
        assert group['taxids'].map(json.loads).tolist() == [['sp000001']] * len(group)
    assert source_bytes(source) == originals
    assert source_bytes(root / 'raw') == before
    first = path.read_bytes()
    assert merge_merops(root).read_bytes() == first


def test_csv_rejects_changed_registered_files_and_raw_output(snapshot):
    source, root = snapshot
    merops.acquire(root / 'raw/merops', source_root=source)
    with pytest.raises(ValueError, match='outside raw'):
        merge_merops(root, output_path=root / 'raw/merged.csv')
    path = merge_merops(root)
    first = path.read_bytes()
    (root / 'raw/merops/03_source/matrices.pt').write_bytes(b'Changed opaque source bytes')
    with pytest.raises(ValueError, match='changed'):
        merge_merops(root)
    assert path.read_bytes() == first


def test_missing_original_dictionaries_are_reported(snapshot):
    source, root = snapshot
    with pytest.raises(FileNotFoundError, match='Original MEROPS dictionaries'):
        merops.acquire(root / 'raw/merops', source_root=source / 'missing')
    with pytest.raises(ValueError, match='outside the original'):
        merops.acquire(source / 'nested/raw/merops', source_root=source)
