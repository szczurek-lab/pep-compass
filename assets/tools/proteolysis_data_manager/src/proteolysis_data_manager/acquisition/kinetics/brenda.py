"""Register the unchanged official BRENDA JSON archive obtained after licensing."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

from proteolysis_data_manager.acquisition.helpers.raw import acquire_files
from proteolysis_data_manager.utils.io import sha256
from proteolysis_data_manager.utils.paths import repository_root
from proteolysis_data_manager.utils.storage import read_download_receipt

OFFICIAL_URL = 'https://www.brenda-enzymes.org/download.php'


def default_root():
    """Return the default kinetic-data series root.

    :return: Checkout-local kinetics directory.
    :rtype: pathlib.Path
    """
    return repository_root() / 'data/proteolysis/kinetics_datasets'


def archive_metadata(path):
    """Check the archive and read the official JSON header with bounded memory.

    :param path: Published BRENDA tar archive.
    :return: JSON member name, release and JSON schema version.
    :rtype: dict
    :raises ValueError: If the archive or required JSON header is invalid.
    """
    import ijson

    with tarfile.open(path, 'r:*') as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith('.json')]
        if len(members) != 1:
            raise ValueError('Expected exactly one JSON member in the BRENDA archive')
        header, data_seen = {}, False
        with archive.extractfile(members[0]) as handle:
            for prefix, event, value in ijson.parse(handle):
                if prefix in {'release', 'version'} and event == 'string':
                    header[prefix] = value
                elif prefix == 'data' and event == 'start_map':
                    data_seen = True
                if len(header) == 2 and data_seen:
                    break
        if len(header) != 2 or not data_seen:
            raise ValueError('Expected BRENDA release/version/data JSON structure')
    return {'json_member': members[0].name, 'source_version': header['release'], 'json_schema_version': header['version']}


def verify_archive(path):
    """Return the single JSON member after archive/header validation.

    :param path: Published BRENDA archive.
    :return: Member name.
    :rtype: str
    """
    return archive_metadata(path)['json_member']


def read_original_receipt(path, digest):
    """Read a shared, historical or installer receipt covering this archive.

    :param path: Original receipt JSON path.
    :param str digest: SHA-256 checksum of the supplied unchanged archive.
    :return: Verified original file receipt with its known retrieval time.
    :rtype: dict
    :raises ValueError: If the receipt does not cover the archive checksum.
    """
    original = json.loads(Path(path).read_text())
    source = original.get('sources', {}).get('brenda', original)
    entries = source.get('files')
    if isinstance(entries, dict):
        candidates = entries.values()
    elif isinstance(entries, list):
        candidates = [entry | {'retrieved_at': source.get('retrieved_at')} for entry in entries]
    else:
        candidates = [source | {'retrieved_at': source.get('retrieved_at', source.get('retrieved_utc'))}]
    for entry in candidates:
        if entry.get('sha256') == digest:
            return {'sha256': digest, 'retrieved_at': entry.get('retrieved_at')}
    raise ValueError('Original receipt does not cover the supplied archive checksum')


def acquire(directory: Path, *, force=False, local_files=None, provenance=None,
            archive=None, receipt=None, release=None):
    """Import the licensed raw BRENDA archive with the common acquisition API.

    :param pathlib.Path directory: Destination ``raw/brenda`` directory.
    :param bool force: Permit replacing an existing registered snapshot.
    :param local_files: Optional single filename/path mapping for a local archive.
    :param provenance: Optional original per-file receipt mapping.
    :param archive: Local archive downloaded through the official BRENDA page.
    :param receipt: Optional original receipt or installer provenance JSON.
    :param release: Required published release, if specified.
    :return: Source receipt with the original archive and header metadata.
    :rtype: dict
    :raises ValueError: If provenance, release or archive validation fails.
    :raises FileNotFoundError: If no licensed archive was supplied.
    """
    directory = Path(directory)
    local_files, provenance = dict(local_files or {}), dict(provenance or {})
    if archive is not None:
        archive = Path(archive)
        local_files[archive.name] = archive
    if not local_files:
        previous = read_download_receipt(directory.parent.parent, 'brenda', missing_ok=True)
        local_files = {name: directory / name for name in previous.get('files', {})}
    if len(local_files) != 1:
        raise FileNotFoundError(f'Supply one licensed BRENDA JSON archive from {OFFICIAL_URL}')
    name, path = next(iter(local_files.items()))
    metadata = archive_metadata(path)
    if release is not None and metadata['source_version'] != release:
        raise ValueError(f'BRENDA release {metadata["source_version"]!r} differs from required {release!r}')
    digest = sha256(path)
    if receipt is not None:
        provenance[name] = read_original_receipt(receipt, digest)

    previous = read_download_receipt(directory.parent.parent, 'brenda', missing_ok=True)
    if previous.get('files'):
        recorded = previous['files'].get(name)
        if recorded is None or (recorded['sha256'] != digest and not force):
            raise ValueError('A different BRENDA snapshot already exists; use a separate data root')
        if recorded['sha256'] == digest:
            provenance.setdefault(name, recorded)

    def validate(files):
        candidate = archive_metadata(files[name])
        if candidate != metadata:
            raise ValueError('BRENDA archive metadata changed during import')

    return acquire_files(directory, source='brenda', files={name: OFFICIAL_URL},
                         version=metadata['source_version'], validate=validate,
                         force=force, local_files=local_files, provenance=provenance)


def import_snapshot(archive, root, *, receipt=None, release=None):
    """Retain the preceding import interface using the shared raw manifest.

    :param archive: Licensed local BRENDA archive.
    :param root: Kinetics data root.
    :param receipt: Original acquisition receipt.
    :param release: Required BRENDA release.
    :return: Backwards-compatible source receipt.
    :rtype: dict
    """
    acquire(Path(root) / 'raw/brenda', archive=archive, receipt=receipt, release=release)
    return verify_snapshot(root)


def verify_snapshot(root):
    """Verify a raw BRENDA snapshot and expose the existing reader interface.

    :param root: Kinetics data root.
    :return: Verified archive location, checksum, release and import metadata.
    :rtype: dict
    :raises ValueError: If the archive changed after acquisition.
    """
    root = Path(root)
    source = read_download_receipt(root, 'brenda')
    if 'raw_path' in source:
        path = root / source['raw_path']
        if sha256(path) != source['sha256']:
            raise ValueError('BRENDA raw archive changed after acquisition')
        return source
    if len(source['files']) != 1:
        raise ValueError('Expected one acquired BRENDA archive')
    name, entry = next(iter(source['files'].items()))
    path = root / 'raw/brenda' / name
    if sha256(path) != entry['sha256']:
        raise ValueError('BRENDA raw archive changed after acquisition')
    return entry | archive_metadata(path) | {'source': 'brenda', 'raw_path': str(path.relative_to(root))}
