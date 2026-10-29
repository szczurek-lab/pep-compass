"""Raw byte preservation, source routing, snapshot validation and provenance."""
import ast
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
from openpyxl import Workbook

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets/tools/proteolysis_data_manager/src'))

from proteolysis_data_manager.acquisition.helpers import raw
from proteolysis_data_manager.acquisition.half_life import peplife2, pepmsnd, dramp, plifepred, cavaco
from proteolysis_data_manager.acquisition.half_life.download_all import acquire_all
from proteolysis_data_manager.acquisition.kinetics import brenda
from proteolysis_data_manager.utils.io import sha256
from proteolysis_data_manager.utils.paths import default_data_root


def test_half_life_default_root_is_in_proteolysis_directory():
    assert default_data_root() == REPO / 'data/proteolysis/half_time_datasets'


def test_kinetics_default_root_is_in_proteolysis_directory():
    assert brenda.default_root() == REPO / 'data/proteolysis/kinetics_datasets'


@pytest.fixture
def api_files():
    def payload(identifier):
        return json.dumps({'status': 200, 'count': 1, 'data': [{'id': identifier, 'half_life': '<30', 'units_half': 'minutes'}]}).encode()
    return {'natural.json': payload('1'), 'modified.json': payload('2'),
            'complete.fasta': b'>1\nAAAA\n>2\nCCCC\n'}


@pytest.fixture
def network(monkeypatch, api_files):
    requests = []
    def retrieve(request, timeout):
        requests.append(request.full_url)
        name = next(name for name, url in peplife2.FILES.items() if url == request.full_url)
        return io.BytesIO(api_files[name])
    monkeypatch.setattr(raw, 'urlopen', retrieve)
    return requests


def test_raw_api_bytes_and_original_receipt_survive_rerun(tmp_path, api_files, network):
    folder = tmp_path / 'raw/peplife2'
    first = peplife2.acquire(folder)
    manifest = tmp_path / 'raw/manifest.json'
    original = manifest.read_bytes()
    assert peplife2.acquire(folder) == first
    assert manifest.read_bytes() == original
    assert len(network) == 3
    assert list(tmp_path.rglob('*manifest*.json')) == [manifest]
    for name, content in api_files.items():
        assert (folder / name).read_bytes() == content
        assert first['files'][name]['status'] == 'downloaded'
        assert first['files'][name]['retrieved_at']
    (folder / 'natural.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        peplife2.acquire(folder)


def test_forced_incomplete_snapshot_preserves_previous_files_and_manifest(tmp_path, api_files, network):
    folder = tmp_path / 'raw/peplife2'
    peplife2.acquire(folder)
    manifest = tmp_path / 'raw/manifest.json'
    original = manifest.read_bytes()
    expected = {name: (folder / name).read_bytes() for name in api_files}
    api_files['natural.json'] = b'{"status":200,"count":1,"data":[{"id":"3"}]}'
    with pytest.raises(ValueError, match='differs'):
        peplife2.acquire(folder, force=True)
    assert manifest.read_bytes() == original
    assert expected == {name: (folder / name).read_bytes() for name in api_files}
    assert not list((tmp_path / 'raw').glob('.peplife2-*'))


def test_failed_http_response_never_registers_a_source(tmp_path, api_files, network):
    api_files['natural.json'] = b'{"status":404,"count":0,"data":[]}'
    with pytest.raises(ValueError, match='Invalid'):
        peplife2.acquire(tmp_path / 'raw/peplife2')
    assert not (tmp_path / 'raw/manifest.json').exists()
    assert not (tmp_path / 'raw/peplife2/natural.json').exists()


@pytest.mark.parametrize('module,name,headers', [
    (dramp, 'stability_amps.xlsx', ['DRAMP_ID', 'Sequence', 'half_life', 'Type']),
    (pepmsnd, 'dataset.xlsx', ['ID', 'Sequence', 'T1/2(hour)', 'Test Sample', 'In-Vivo/Vitro']),
])
def test_published_workbooks_remain_unchanged(tmp_path, module, name, headers, monkeypatch):
    original = tmp_path / name
    book = Workbook(); book.active.append(headers); book.active.append(['raw value'] * len(headers)); book.save(original)
    monkeypatch.setattr(raw, 'urlopen', lambda *args, **kwargs: pytest.fail('Local import used the network'))
    receipt = module.acquire(tmp_path / 'data/raw' / module.__name__.rsplit('.', 1)[-1], local_files={name: original})
    entry = receipt['files'][name]
    assert entry['sha256'] == sha256(original)
    assert entry['status'] == 'imported_local' and entry['retrieved_at'] is None
    book.active.delete_cols(1); book.save(original)
    with pytest.raises(ValueError, match='required source columns missing'):
        module.acquire(tmp_path / 'invalid/raw/source', local_files={name: original})


def test_orchestrator_routes_each_source_to_its_own_directory(tmp_path, monkeypatch):
    selected = ['peplife2', 'pepmsnd', 'plifepred', 'dramp', 'cavaco']
    calls = []
    def source_acquire(directory, **options):
        calls.append((directory, options))
        return {'source': directory.name}
    for module in [peplife2, pepmsnd, dramp, plifepred, cavaco]:
        monkeypatch.setattr(module, 'acquire', source_acquire)
    assert list(acquire_all(tmp_path)) == selected
    assert [path for path, _ in calls] == [tmp_path / 'raw' / source for source in selected]


def test_multiple_sources_share_one_receipt_file(tmp_path, api_files, network):
    peplife2.acquire(tmp_path / 'raw/peplife2')
    workbook = tmp_path / 'dataset.xlsx'
    book = Workbook(); book.active.append(['ID', 'Sequence', 'T1/2(hour)', 'Test Sample', 'In-Vivo/Vitro']); book.save(workbook)
    pepmsnd.acquire(tmp_path / 'raw/pepmsnd', local_files={'dataset.xlsx': workbook})
    manifest = json.loads((tmp_path / 'raw/manifest.json').read_text())
    assert set(manifest['sources']) == {'peplife2', 'pepmsnd'}
    assert len(list(tmp_path.rglob('*manifest*.json'))) == 1
    assert manifest['sources']['peplife2']['files']['natural.json']['sha256'] == sha256(tmp_path / 'raw/peplife2/natural.json')


def test_existing_source_receipt_is_imported_into_shared_manifest_without_retrieval(tmp_path, network):
    folder = tmp_path / 'raw/peplife2'
    first = peplife2.acquire(folder)
    manifest = tmp_path / 'raw/manifest.json'
    legacy = tmp_path / 'manifests/download_peplife2.json'
    legacy.parent.mkdir()
    legacy.write_text(json.dumps(first))
    manifest.unlink()
    assert peplife2.acquire(folder) == first
    assert len(network) == 3
    assert json.loads(manifest.read_text())['sources']['peplife2'] == first
    assert legacy.exists()  # Existing historical metadata is not silently deleted.


@pytest.fixture
def brenda_archive(tmp_path):
    path = tmp_path / 'official.tar.gz'
    raw = json.dumps({'release': '2026.1', 'version': '2.0.0', 'data': {'3.4.1.1': {'untouched': ['raw']}}}).encode()
    with tarfile.open(path, 'w:gz') as archive:
        member = tarfile.TarInfo('brenda.json'); member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    return path


def test_brenda_import_keeps_archive_and_true_retrieval_time(tmp_path, brenda_archive, monkeypatch):
    receipt = tmp_path / 'provenance.json'
    receipt.write_text(json.dumps({'sha256': sha256(brenda_archive), 'retrieved_utc': '2026-09-17T09:04:50+00:00'}))
    root = tmp_path / 'kinetics'
    monkeypatch.setattr(raw, 'urlopen', lambda *a, **k: pytest.fail('BRENDA import used the network'))
    result = brenda.acquire(root / 'raw/brenda', archive=brenda_archive, receipt=receipt, release='2026.1')
    entry = result['files'][brenda_archive.name]
    assert entry['status'] == 'imported_local'
    assert entry['retrieved_at'] == '2026-09-17T09:04:50+00:00'
    assert (root / 'raw/brenda' / brenda_archive.name).read_bytes() == brenda_archive.read_bytes()
    before = (root / 'raw/manifest.json').read_bytes()
    assert brenda.acquire(root / 'raw/brenda') == result
    assert (root / 'raw/manifest.json').read_bytes() == before
    assert brenda.verify_snapshot(root)['source_version'] == '2026.1'
    with pytest.raises(ValueError, match='release'):
        brenda.acquire(tmp_path / 'wrong/raw/brenda', archive=brenda_archive, release='2025.1')
    receipt.write_text('{"sha256":"wrong"}')
    with pytest.raises(ValueError, match='checksum'):
        brenda.acquire(tmp_path / 'invalid/raw/brenda', archive=brenda_archive, receipt=receipt)


def test_acquisition_has_no_adapter_or_integration_imports():
    folder = REPO / 'assets/tools/proteolysis_data_manager/src/proteolysis_data_manager/acquisition'
    for path in folder.rglob('*.py'):
        tree = ast.parse(path.read_text())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): modules.extend(alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom): modules.append(node.module or '')
        assert not any('.integration' in name or '.adapters' in name for name in modules), path


def test_cli_routes_all_types_through_the_shared_orchestrator(tmp_path, monkeypatch):
    from proteolysis_data_manager.acquisition import download_all
    from proteolysis_data_manager.cli import download_raw
    calls = []
    monkeypatch.setattr(download_all, 'acquire_all', lambda **options: calls.append(options))
    assert download_raw(['--type', 'all', '--source', 'dramp', '--half-life-root', str(tmp_path),
                         '--brenda-archive', 'official.tar.gz', '--brenda-release', '2026.1']) == 0
    assert calls == [dict(half_life_root=tmp_path, kinetics_root=None, brenda_archive=Path('official.tar.gz'),
                          brenda_receipt=None, brenda_release='2026.1', force=False,
                          data_types=('half_life', 'window_neighbourhood', 'kinetics'), sources=['dramp'],
                          window_root=None, window_sources=None, kinetics_sources=None)]


def test_half_life_cli_passes_explicit_data_root(tmp_path, monkeypatch):
    import proteolysis_data_manager.acquisition.half_life as half_life
    from proteolysis_data_manager.cli import download_half_life

    calls = []
    monkeypatch.setattr(half_life, 'acquire', lambda source, root, **options: calls.append((source, root)))
    assert download_half_life(['--data-root', str(tmp_path), '--source', 'peplife2']) == 0
    assert calls == [('peplife2', tmp_path)]
