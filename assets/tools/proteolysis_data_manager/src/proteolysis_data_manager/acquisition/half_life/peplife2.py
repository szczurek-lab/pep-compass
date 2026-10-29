"""Acquire the published peplife2 artifacts without adapting their records."""
from pathlib import Path
from proteolysis_data_manager.acquisition.helpers.raw import acquire_files, validate_container

FILES = {'natural.json': 'https://webs.iiitd.edu.in/raghava/peplife2/api/api.php?dataType=seq&dataValue=Natural',
 'modified.json': 'https://webs.iiitd.edu.in/raghava/peplife2/api/api.php?dataType=seq&dataValue=Modified',
 'complete.fasta': 'https://webs.iiitd.edu.in/raghava/peplife2/dataset/Peplife_complete.fasta'}
VERSION = 'unversioned_public_snapshot'


def validate(files):
    """Validate the published source snapshot without transforming it.

    :param files: Original filenames mapped to local paths.
    :raises ValueError: If a file has an invalid source format.
    """
    records(files)


def validate_response(path):
    """Check API status, count and unique source IDs.

    :param pathlib.Path path: Unmodified API response.
    :return: Original response records.
    :rtype: list[dict]
    :raises ValueError: If the API response is incomplete or contains an error.
    """
    import json
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or str(payload.get('status')) != '200' or not isinstance(payload.get('data'), list):
        raise ValueError(f'Invalid PEPlife2 response: {path}')
    identifiers = [str(record['id']) for record in payload['data']]
    if int(payload.get('count', -1)) != len(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f'PEPlife2 count or identifiers do not match: {path}')
    return payload['data']


def records(files):
    """Verify that natural/modified API IDs exactly cover complete FASTA.

    :param files: Original filenames mapped to local paths.
    :return: Unchanged API records from both disjoint partitions.
    :rtype: list[dict]
    :raises ValueError: If source IDs repeat or completeness verification fails.
    """
    by_id = {}
    for name in ('natural.json', 'modified.json'):
        for record in validate_response(files[name]):
            identifier = str(record['id'])
            if identifier in by_id:
                raise ValueError(f'Duplicate PEPlife2 ID across partitions: {identifier}')
            by_id[identifier] = record
    validate_container(files['complete.fasta'])
    fasta_ids = [line[1:].strip() for line in Path(files['complete.fasta']).read_text().splitlines() if line.startswith('>')]
    if len(set(fasta_ids)) != len(fasta_ids) or set(fasta_ids) != set(by_id):
        raise ValueError('PEPlife2 JSON union differs from complete FASTA identifiers')
    return list(by_id.values())


def acquire(directory: Path, *, force=False, local_files=None, provenance=None):
    """Store unchanged raw peplife2 artifacts and update the shared receipt.

    :param pathlib.Path directory: Destination ``raw/peplife2`` directory.
    :param bool force: Replace existing registered source artifacts.
    :param local_files: Optional mapping of original local artifacts to import.
    :param provenance: Original per-file receipts for imported artifacts.
    :return: Source receipt stored within ``raw/manifest.json``.
    :rtype: dict
    :raises ValueError: If source validation or checksum verification fails.
    :raises OSError: If retrieval or storage fails.
    """
    return acquire_files(directory, source='peplife2', files=FILES, version=VERSION,
                         validate=validate, force=force, local_files=local_files, provenance=provenance)
