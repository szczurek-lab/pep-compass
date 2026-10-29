"""Acquire the published plifepred artifacts without adapting their records."""
from pathlib import Path
from proteolysis_data_manager.acquisition.helpers.raw import acquire_files, validate_workbook

FILES = {'natural.xlsx': 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0196829.s008&type=supplementary',
 'mixed.xlsx': 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0196829.s007&type=supplementary'}
VERSION = 'unversioned_public_snapshot'


def validate(files):
    """Validate the published source snapshot without transforming it.

    :param files: Original filenames mapped to local paths.
    :raises ValueError: If a file has an invalid source format.
    """
    validate_workbook(files['natural.xlsx'], {'id.pdb', 'seq', 'hl in seconds'})
    validate_workbook(files['mixed.xlsx'], {'id.pdb', 'seq', 'Half-life in seconds'})


def acquire(directory: Path, *, force=False, local_files=None, provenance=None):
    """Store unchanged raw plifepred artifacts and update the shared receipt.

    :param pathlib.Path directory: Destination ``raw/plifepred`` directory.
    :param bool force: Replace existing registered source artifacts.
    :param local_files: Optional mapping of original local artifacts to import.
    :param provenance: Original per-file receipts for imported artifacts.
    :return: Source receipt stored within ``raw/manifest.json``.
    :rtype: dict
    :raises ValueError: If source validation or checksum verification fails.
    :raises OSError: If retrieval or storage fails.
    """
    return acquire_files(directory, source='plifepred', files=FILES, version=VERSION,
                         validate=validate, force=force, local_files=local_files, provenance=provenance)
