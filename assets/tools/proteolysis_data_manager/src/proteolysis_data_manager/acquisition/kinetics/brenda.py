"""Import and verify a locally obtained BRENDA release without changing raw."""
from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from proteolysis_data_manager.utils.io import sha256, write_json
from proteolysis_data_manager.utils.paths import repository_root


def default_root():
    return repository_root() / 'data/kinetics_datasets'


def verify_archive(path):
    with tarfile.open(path, 'r:*') as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith('.json')]
        if len(members) != 1:
            raise ValueError('Expected exactly one JSON member in the BRENDA archive')
        return members[0].name


def import_snapshot(archive, root, *, receipt=None, release=None):
    """Import once; preserve the original retrieval receipt and exact bytes."""
    archive, root = Path(archive), Path(root)
    digest = sha256(archive)
    member = verify_archive(archive)
    original = json.loads(Path(receipt).read_text()) if receipt else None
    if original:
        entries = original.get('files', [])
        if not any(e.get('sha256') == digest for e in entries):
            raise ValueError('Original receipt does not cover the supplied archive checksum')
        release = release or original.get('source_version_or_accession')
    folder = root / 'raw/brenda'
    target = folder / archive.name
    manifest_path = root / 'manifests/download_brenda.json'
    if manifest_path.exists():
        previous = verify_snapshot(root)
        if previous['sha256'] != digest:
            raise ValueError('A different BRENDA snapshot already exists; use a separate data root')
        return previous
    folder.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) != digest:
        raise ValueError('An unreceipted different raw archive already exists')
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                with archive.open('rb') as origin:
                    shutil.copyfileobj(origin, handle)
                handle.flush()
                if sha256(temporary) != digest:
                    raise ValueError('Archive changed during import')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    manifest = {
        'source': 'brenda', 'source_version': release,
        'status': 'imported_local', 'raw_path': str(target.relative_to(root)),
        'json_member': member, 'sha256': digest, 'bytes': target.stat().st_size,
        'imported_from': str(archive.resolve()), 'imported_at': datetime.now(UTC).isoformat(),
        'retrieved_at': original.get('retrieved_at') if original else None,
        'original_receipt': original,
    }
    write_json(manifest_path, manifest)
    return manifest


def verify_snapshot(root):
    root = Path(root)
    manifest = json.loads((root / 'manifests/download_brenda.json').read_text())
    path = root / manifest['raw_path']
    if sha256(path) != manifest['sha256']:
        raise ValueError('BRENDA raw archive changed after acquisition')
    return manifest
