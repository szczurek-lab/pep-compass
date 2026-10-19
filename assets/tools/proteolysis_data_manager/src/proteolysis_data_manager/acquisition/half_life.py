"""Canonical acquisition of peptide half-life sources, independent of notebooks."""
from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from proteolysis_data_manager.utils.io import sha256, write_json
from proteolysis_data_manager.utils.storage import download_manifest
from proteolysis_data_manager.utils.paths import default_data_root
PEPMSND_COMMIT = 'c0fa6957c7f88a57af29410cfc075542c92a2c63'
SOURCES = {
    'peplife2': {
        'natural.json': 'https://webs.iiitd.edu.in/raghava/peplife2/api/api.php?dataType=seq&dataValue=Natural',
        'modified.json': 'https://webs.iiitd.edu.in/raghava/peplife2/api/api.php?dataType=seq&dataValue=Modified',
        'complete.fasta': 'https://webs.iiitd.edu.in/raghava/peplife2/dataset/Peplife_complete.fasta',
    },
    'pepmsnd': {'dataset.xlsx': f'https://raw.githubusercontent.com/hmenghu/PepMSND/{PEPMSND_COMMIT}/Dataset/Dataset.xlsx'},
    'plifepred': {
        'natural.xlsx': 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0196829.s008&type=supplementary',
        'mixed.xlsx': 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0196829.s007&type=supplementary',
    },
    'dramp': {'stability_amps.xlsx': 'https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/stability_amps.xlsx'},
    'cavaco': {
        'article.xml': 'https://pmc-oa-opendata.s3.amazonaws.com/PMC8301568.1/PMC8301568.1.xml',
        'article.pdf': 'https://pmc-oa-opendata.s3.amazonaws.com/PMC8301568.1/PMC8301568.1.pdf',
        'supplement.docx': 'https://pmc-oa-opendata.s3.amazonaws.com/PMC8301568.1/CTS-14-1349-s001.docx',
        'supplement.pdf': 'https://pmc-oa-opendata.s3.amazonaws.com/PMC8301568.1/CTS-14-1349-s003.pdf',
    },
}
# Explicit file-name mapping for importing the preceding read-only audit.
AUDIT_FILES = {
    ('peplife2', 'natural.json'): 'peplife_natural.json',
    ('peplife2', 'modified.json'): 'peplife_modified.json',
    ('peplife2', 'complete.fasta'): 'peplife_fasta.txt',
    ('pepmsnd', 'dataset.xlsx'): 'pepmsnd.xlsx',
    ('plifepred', 'natural.xlsx'): 'plifepred_natural.xlsx',
    ('plifepred', 'mixed.xlsx'): 'plifepred_modified.xlsx',
    ('cavaco', 'article.xml'): 'PMC8301568.1.xml',
    ('cavaco', 'article.pdf'): 'PMC8301568.1.pdf',
    ('cavaco', 'supplement.docx'): 'CTS-14-1349-s001.docx',
    ('cavaco', 'supplement.pdf'): 'CTS-14-1349-s003.pdf',
}


def validate_file(path: Path) -> None:
    if path.suffix == '.json':
        payload = json.loads(path.read_text())
        if str(payload.get('status')) != '200' or not isinstance(payload.get('data'), list):
            raise ValueError(f'Invalid PEPlife2 response: {path}')
        if int(payload.get('count', -1)) != len(payload['data']):
            raise ValueError(f'PEPlife2 count does not match rows: {path}')
    elif path.suffix in {'.xlsx', '.docx'}:
        with ZipFile(path) as archive:
            required = 'xl/workbook.xml' if path.suffix == '.xlsx' else 'word/document.xml'
            if required not in archive.namelist():
                raise ValueError(f'Unexpected spreadsheet/document response: {path}')
    elif path.suffix == '.pdf' and not path.read_bytes().startswith(b'%PDF-'):
        raise ValueError(f'Expected PDF: {path}')
    elif path.suffix == '.xml':
        import xml.etree.ElementTree as ET
        if ET.parse(path).getroot().tag != 'article':
            raise ValueError(f'Expected JATS article: {path}')
    elif path.suffix == '.fasta' and not path.read_text().startswith('>'):
        raise ValueError(f'Expected FASTA: {path}')


def acquire(source: str, root: Path | None = None, *, force: bool = False,
            import_audit: Path | None = None, import_legacy: Path | None = None) -> None:
    """Keep verified raw files and their original receipt on reruns.

    Audit/legacy imports are marked as imports, never as fresh network retrievals.
    An unreceipted or modified file is not silently accepted as the recorded source.
    """
    root = Path(root) if root is not None else default_data_root()
    folder = root / 'raw' / source
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = download_manifest(root,source)
    existing_manifest = download_manifest(root,source,legacy_fallback=True)
    manifest = json.loads(existing_manifest.read_text()) if existing_manifest.exists() else {'source': source, 'files': {}}
    for name, url in SOURCES[source].items():
        target = folder / name
        previous = manifest['files'].get(name)
        if target.exists() and not force:
            validate_file(target)
            if not previous or previous['sha256'] != sha256(target) or previous.get('requested_url') != url:
                raise ValueError(f'File is unreceipted or changed: {target}; use --force to reacquire')
            continue
        imported = None
        if import_audit and (source, name) in AUDIT_FILES:
            candidate = import_audit / AUDIT_FILES[source, name]
            if candidate.is_file():
                imported = candidate
        if source == 'dramp' and import_legacy:
            candidate = import_legacy / 'raw' / 'dramp' / name
            if candidate.is_file():
                imported = candidate
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                if imported:
                    with imported.open('rb') as origin:
                        shutil.copyfileobj(origin, handle)
                else:
                    request = Request(url, headers={'User-Agent': 'pep-compass-half-life/1.0'})
                    with urlopen(request, timeout=60) as response:
                        shutil.copyfileobj(response, handle)
                handle.flush()
                # Format validation uses the actual suffix before replacing any artifact.
                formatted = temporary.with_suffix(target.suffix)
                temporary.rename(formatted)
                temporary = formatted
                validate_file(temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        receipt = {'requested_url': url, 'sha256': sha256(target), 'bytes': target.stat().st_size}
        now = datetime.now(UTC).isoformat()
        if imported:
            receipt.update(status='imported_local', imported_at=now, imported_from=str(imported.resolve()))
            receipt['retrieved_at'] = None
            if import_legacy and source == 'dramp':
                old = import_legacy / 'raw' / 'dramp' / 'download_manifest.json'
                if old.is_file():
                    old_payload = json.loads(old.read_text())
                    receipt['original_manifest'] = old_payload
                    receipt['retrieved_at'] = old_payload.get('retrieved_at')
            if import_audit:
                audit = json.loads((Path(__file__).parent / 'resources/audit_snapshot.json').read_text())
                if any(f.get('sha256') == receipt['sha256'] for f in audit['files']):
                    receipt['audit_date'] = audit['audit_date']
        else:
            receipt.update(status='downloaded', retrieved_at=now)
        manifest['files'][name] = receipt
        manifest['version'] = PEPMSND_COMMIT if source == 'pepmsnd' else ('PMC8301568.1' if source == 'cavaco' else 'unversioned_public_snapshot')
        write_json(manifest_path, manifest)
        print(f"{receipt['status']}: {target}")
    if source == 'peplife2':
        peplife_records(root)  # Verify completeness even when reusing local files.
    if not manifest_path.exists(): write_json(manifest_path,manifest)
    if existing_manifest!=manifest_path: existing_manifest.unlink(missing_ok=True)


def peplife_records(root: Path | None = None) -> list[dict]:
    root = Path(root) if root is not None else default_data_root()
    folder = root / 'raw' / 'peplife2'
    by_id = {}
    for name in ('natural.json', 'modified.json'):
        validate_file(folder / name)
        for row in json.loads((folder / name).read_text())['data']:
            identifier = str(row['id'])
            if identifier in by_id:
                raise ValueError(f'Duplicate PEPlife2 ID across partitions: {identifier}')
            by_id[identifier] = row
    fasta_ids = [line[1:].strip() for line in (folder / 'complete.fasta').read_text().splitlines() if line.startswith('>')]
    if len(fasta_ids) != len(set(fasta_ids)) or set(fasta_ids) != set(by_id):
        raise ValueError('PEPlife2 JSON union differs from complete FASTA identifiers')
    return list(by_id.values())


def verify_raw_sources(root: Path | None = None):
    """Verify the local snapshot before any adapter reads source records."""
    root = Path(root) if root is not None else default_data_root()
    for source, files in SOURCES.items():
        manifest = json.loads(download_manifest(root, source, legacy_fallback=True).read_text())
        for name in files:
            entry = manifest['files'][name]
            path = root / 'raw' / source / name
            if sha256(path) != entry['sha256']:
                raise ValueError(f'Source changed after acquisition: {source}/{name}')
    peplife_records(root)
