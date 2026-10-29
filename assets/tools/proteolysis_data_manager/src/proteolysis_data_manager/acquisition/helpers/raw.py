"""Store unchanged source bytes and one receipt per raw-data series."""
from __future__ import annotations

import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from proteolysis_data_manager.utils.io import sha256, write_json
from proteolysis_data_manager.utils.logger import get_custom_logger
from proteolysis_data_manager.utils.storage import read_download_receipt

logger = get_custom_logger(__name__)


def fetch_bytes(url: str) -> bytes:
    """Read one original API response for source discovery.

    :param str url: Official source URL.
    :return: Unmodified response bytes.
    :rtype: bytes
    :raises OSError: If the source request fails.
    """
    request = Request(url, headers={'User-Agent': 'pep-compass-raw-acquisition/1.0',
                                    'Accept': 'application/json'})
    with urlopen(request, timeout=60) as response:
        return response.read()


def validate_container(path: Path) -> None:
    """Validate the published container without changing its contents.

    :param pathlib.Path path: Downloaded source artifact.
    :raises ValueError: If the artifact is empty or has an unexpected format.
    """
    if not path.stat().st_size:
        raise ValueError(f'Empty source artifact: {path}')
    if path.suffix in {'.xlsx', '.docx'}:
        required = 'xl/workbook.xml' if path.suffix == '.xlsx' else 'word/document.xml'
        try:
            with ZipFile(path) as archive:
                if required not in archive.namelist() or archive.testzip() is not None:
                    raise ValueError(f'Invalid published workbook/document: {path}')
        except BadZipFile as error:
            logger.error('Source is not a valid published workbook/document: %s', path, exc_info=True)
            raise ValueError(f'Invalid published workbook/document: {path}') from error
    elif path.suffix == '.pdf':
        with path.open('rb') as handle:
            if not handle.read(5).startswith(b'%PDF-'):
                raise ValueError(f'Expected PDF: {path}')
    elif path.suffix == '.xml':
        if ET.parse(path).getroot().tag != 'article':
            raise ValueError(f'Expected JATS article: {path}')
    elif path.suffix == '.fasta':
        with path.open() as handle:
            if not handle.read(1) == '>':
                raise ValueError(f'Expected FASTA: {path}')


def validate_workbook(path: Path, required: set[str]) -> None:
    """Check original worksheet headers without mapping their scientific values.

    :param pathlib.Path path: Original published workbook.
    :param required: Source-specific required column names.
    :raises ValueError: If the active worksheet lacks expected columns.
    """
    from openpyxl import load_workbook

    validate_container(path)
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        headers = next(workbook.active.values, ())
        if missing := required.difference(headers):
            raise ValueError(f'{path}: required source columns missing: {sorted(missing)}')
    finally:
        workbook.close()


def acquire_files(
    directory: Path,
    *,
    source: str,
    files: Mapping[str, str],
    version: str | None,
    validate: Callable[[Mapping[str, Path]], None],
    force: bool = False,
    local_files: Mapping[str, Path] | None = None,
    provenance: Mapping[str, dict] | None = None,
    payloads: Mapping[str, bytes] | None = None,
) -> dict:
    """Acquire and validate a complete source before replacing existing files.

    :param pathlib.Path directory: Source directory immediately inside ``raw``.
    :param str source: Database identifier used in the shared manifest.
    :param files: Mapping of original filenames to source URLs.
    :param version: Published release or pinned revision, if available.
    :param validate: Source-specific validation of the complete candidate snapshot.
    :param bool force: Permit replacing an existing registered artifact.
    :param local_files: Optional original files to copy without modification.
    :param provenance: Optional original receipt fields indexed by filename.
    :param payloads: Unmodified responses already fetched during source discovery.
    :return: Source receipt within ``raw/manifest.json``.
    :rtype: dict
    :raises ValueError: If recorded bytes changed or source validation fails.
    :raises OSError: If downloading or storing a source fails.
    """
    directory = Path(directory)
    root = directory.parent.parent
    manifest_path = directory.parent / 'manifest.json'
    previous = read_download_receipt(root, source, missing_ok=True)
    entries = {name: entry for name, entry in previous.get('files', {}).items() if name in files}
    local_files, provenance = local_files or {}, provenance or {}
    payloads = payloads or {}
    directory.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Acquiring raw source %s', source)

    # Stage unchanged source bytes and retain original receipts on reruns.
    with tempfile.TemporaryDirectory(prefix=f'.{source}-', dir=directory.parent) as temporary:
        staged = {}
        candidates = {}
        for name, url in files.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError(f'Source filenames must be relative paths: {name}')
            target = directory / name
            recorded = entries.get(name)
            if target.exists() and not force:
                if not recorded or recorded['sha256'] != sha256(target) or recorded.get('requested_url') != url:
                    raise ValueError(f'Raw file is unreceipted or changed: {target}; use --force to reacquire')
                candidates[name] = target
                continue
            candidate = Path(temporary) / name
            candidate.parent.mkdir(parents=True, exist_ok=True)
            imported = local_files.get(name)
            logger.debug('Acquiring %s from %s', name, imported or url)
            if name in payloads:
                if imported is not None:
                    raise ValueError(f'Both a network response and local import were supplied: {name}')
                candidate.write_bytes(payloads[name])
            elif imported is not None:
                shutil.copyfile(imported, candidate)
            else:
                request = Request(url, headers={'User-Agent': 'pep-compass-raw-acquisition/1.0'})
                with urlopen(request, timeout=60) as response, candidate.open('wb') as output:
                    shutil.copyfileobj(response, output)
            digest = sha256(candidate)
            origin = provenance.get(name, {})
            if origin.get('sha256') and origin['sha256'] != digest:
                raise ValueError(f'Original receipt checksum differs: {name}')
            now = datetime.now(UTC).isoformat()
            entry = {'requested_url': url, 'sha256': digest, 'bytes': candidate.stat().st_size,
                     'status': 'imported_local' if imported is not None else 'downloaded',
                     'retrieved_at': origin.get('retrieved_at') if imported is not None else now}
            if imported is not None:
                entry.update(imported_from=str(Path(imported).resolve()), imported_at=now)
            entries[name] = entry
            candidates[name] = candidate
            staged[name] = candidate

        # Source validation completes before any previous artifact is replaced.
        validate(candidates)
        receipt = {'source': source, 'version': version, 'files': entries}
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'version': 1, 'sources': {}}
        manifest['sources'][source] = receipt
        directory.mkdir(parents=True, exist_ok=True)
        for name, candidate in staged.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            candidate.replace(target)
        if not manifest_path.exists() or manifest != json.loads(manifest_path.read_text()):
            write_json(manifest_path, manifest)
    logger.info('Raw source %s ready in %s', source, directory)
    return receipt
