"""Raw source preservation and routing for cleavage-neighbourhood datasets."""
import io
import json
import sys
from pathlib import Path

import pytest
from openpyxl import Workbook
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'assets/tools/proteolysis_data_manager/src'))
from proteolysis_data_manager.acquisition.helpers import raw
from proteolysis_data_manager.acquisition.window_neighbourhood import merops, mmp_pics, dpp4_qpisa
from proteolysis_data_manager.acquisition.window_neighbourhood.download_all import acquire_all, default_root
from proteolysis_data_manager.utils.io import sha256


@pytest.mark.parametrize('module', [mmp_pics, dpp4_qpisa])
def test_publication_artifacts_are_imported_without_worksheet_mapping(tmp_path, monkeypatch, module):
    originals = tmp_path / 'originals'
    originals.mkdir()
    paths = {}
    for name in module.FILES:
        path = originals / name
        if path.suffix == '.xlsx':
            workbook = Workbook()
            workbook.active.append(['Original multirow header', 'Unspecified ± error'])
            workbook.active.append(['>30', 'NULL'])
            workbook.save(path)
        elif path.suffix == '.xml':
            path.write_text('<article><article-title>Original source</article-title></article>')
        elif path.suffix == '.pdf':
            path.write_bytes(b'%PDF-1.4\nOriginal source paper\n%%EOF\n')
        else:
            with ZipFile(path, 'w') as archive:
                archive.writestr('original_result.txt', 'original result')
        paths[name] = path
    monkeypatch.setattr(raw, 'urlopen', lambda *args, **kwargs: pytest.fail('Local import used network'))
    source = module.__name__.rsplit('.', 1)[-1]
    receipt = module.acquire(tmp_path / 'dataset/raw' / source, local_files=paths)
    for name, original in paths.items():
        assert (tmp_path / 'dataset/raw' / source / name).read_bytes() == original.read_bytes()
        assert receipt['files'][name]['sha256'] == sha256(original)
        assert receipt['files'][name]['retrieved_at'] is None
    manifest = json.loads((tmp_path / 'dataset/raw/manifest.json').read_text())
    assert list(manifest['sources']) == [source]


def test_html_workbook_is_rejected_before_publishing_a_source(tmp_path, monkeypatch):
    monkeypatch.setattr(raw, 'urlopen', lambda *args, **kwargs: io.BytesIO(b'<html>server error</html>'))
    with pytest.raises(ValueError, match='workbook'):
        dpp4_qpisa.acquire(tmp_path / 'raw/dpp4_qpisa')
    assert not (tmp_path / 'raw/manifest.json').exists()


def test_window_orchestrator_owns_the_explicit_series_path(tmp_path, monkeypatch):
    calls = []
    for module in [merops, mmp_pics, dpp4_qpisa]:
        monkeypatch.setattr(module, 'acquire', lambda directory, **options: calls.append(directory))
    acquire_all(tmp_path)
    assert calls == [tmp_path / 'raw' / name for name in ['merops', 'mmp_pics', 'dpp4_qpisa']]
    assert default_root() == ROOT / 'data/proteolysis/window_neighbourhood_datasets'
    with pytest.raises(ValueError, match='Unknown'):
        acquire_all(tmp_path, sources=['unknown'])
