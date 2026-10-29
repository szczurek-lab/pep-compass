"""Complete source pagination and faithful kinetics acquisition receipts."""
import io
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'assets/tools/proteolysis_data_manager/src'))
from proteolysis_data_manager.acquisition.helpers import raw
from proteolysis_data_manager.acquisition.kinetics import brenda, sabio
from proteolysis_data_manager.acquisition.kinetics.download_all import acquire_all


@pytest.fixture
def pages():
    def page(number, identifiers):
        return json.dumps({'meta': {'page': number, 'page_size': 2, 'total_count': 3, 'total_pages': 2},
                           'data': [{'id': i, 'kineticlaw': {'parameter': [{'value': '0.5', 'unit': 'mM'}]}}
                                    for i in identifiers]}, indent=2).encode()
    return {1: page(1, [11, 22]), 2: page(2, [33])}


@pytest.fixture
def network(monkeypatch, pages):
    calls = []
    def retrieve(request, timeout):
        parameters = parse_qs(urlsplit(request.full_url).query)
        assert parameters['q'] == ['ECNumber:3.4.*']
        assert parameters['pageSize'] == ['2']
        number = int(parameters['page'][0])
        calls.append(number)
        return io.BytesIO(pages[number])
    monkeypatch.setattr(raw, 'urlopen', retrieve)
    return calls


def test_all_sabio_pages_are_saved_verbatim_and_reruns_preserve_receipts(tmp_path, pages, network):
    directory = tmp_path / 'raw/sabio'
    first = sabio.acquire(directory, page_size=2)
    manifest = tmp_path / 'raw/manifest.json'
    before = manifest.read_bytes()
    assert network == [1, 2]  # Discovery response is reused, not downloaded twice.
    for number, content in pages.items():
        assert (directory / f'page_{number:06d}.json').read_bytes() == content
        assert first['files'][f'page_{number:06d}.json']['status'] == 'downloaded'
    assert sabio.acquire(directory, page_size=2) == first
    assert network == [1, 2]
    assert manifest.read_bytes() == before
    assert len(list(tmp_path.rglob('*manifest*.json'))) == 1


def test_failed_page_snapshot_never_replaces_valid_raw_data(tmp_path, pages, network):
    directory = tmp_path / 'raw/sabio'
    sabio.acquire(directory, page_size=2)
    manifest = tmp_path / 'raw/manifest.json'
    before = manifest.read_bytes()
    second = (directory / 'page_000002.json').read_bytes()
    changed = json.loads(pages[2])
    changed['data'][0]['id'] = 22
    pages[2] = json.dumps(changed).encode()
    with pytest.raises(ValueError, match='duplicate'):
        sabio.acquire(directory, page_size=2, force=True)
    assert manifest.read_bytes() == before
    assert (directory / 'page_000002.json').read_bytes() == second


@pytest.mark.parametrize('content', [b'<html>new web UI</html>', b'{"error":"service unavailable"}',
                                    b'{"meta":{"page":1,"page_size":2,"total_count":3,"total_pages":2},"data":[]}'])
def test_sabio_rejects_html_api_errors_and_truncated_pages(content):
    with pytest.raises(ValueError):
        sabio.read_page(content)


def test_sabio_rejects_missing_pages_and_changed_totals(tmp_path, pages):
    directory = tmp_path / 'raw'
    directory.mkdir()
    files = {}
    for number, data in pages.items():
        path = directory / f'page_{number:06d}.json'
        path.write_bytes(data)
        files[path.name] = path
    with pytest.raises(ValueError, match='missing'):
        sabio.validate({'page_000001.json': files['page_000001.json']})
    altered = json.loads(pages[2])
    altered['meta']['total_count'] = 4
    altered['data'].append({'id': 44})
    files['page_000002.json'].write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='changed'):
        sabio.validate(files)


def test_sabio_empty_result_is_distinct_from_failed_response(tmp_path, monkeypatch):
    content = b'{"meta":{"page":1,"page_size":1000,"total_count":0,"total_pages":0},"data":[]}'
    monkeypatch.setattr(raw, 'urlopen', lambda *args, **kwargs: io.BytesIO(content))
    sabio.acquire(tmp_path / 'raw/sabio')
    assert (tmp_path / 'raw/sabio/page_000001.json').read_bytes() == content


def test_kinetics_orchestrator_preserves_independent_source_roots(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(brenda, 'acquire', lambda directory, **options: calls.append((directory, options)))
    monkeypatch.setattr(sabio, 'acquire', lambda directory, **options: calls.append((directory, options)))
    acquire_all(tmp_path)
    assert [path for path, _ in calls] == [tmp_path / 'raw/brenda', tmp_path / 'raw/sabio']
    assert calls[1][1]['query'] == 'ECNumber:3.4.*'
    assert brenda.default_root() == ROOT / 'data/proteolysis/kinetics_datasets'


def test_brenda_accepts_original_shared_raw_receipt(tmp_path):
    receipt = tmp_path / 'manifest.json'
    receipt.write_text(json.dumps({'sources': {'brenda': {'files': {'original.tar.gz': {
        'sha256': 'verified-digest', 'retrieved_at': '2026-09-17T09:04:50+00:00'}}}}}))
    assert brenda.read_original_receipt(receipt, 'verified-digest') == {
        'sha256': 'verified-digest', 'retrieved_at': '2026-09-17T09:04:50+00:00'}
    with pytest.raises(ValueError, match='checksum'):
        brenda.read_original_receipt(receipt, 'different')
