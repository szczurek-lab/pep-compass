"""Acquire the published pepmsnd artifacts without adapting their records."""
from pathlib import Path
from proteolysis_data_manager.acquisition.helpers.raw import acquire_files, validate_workbook

FILES = {'dataset.xlsx': 'https://raw.githubusercontent.com/hmenghu/PepMSND/c0fa6957c7f88a57af29410cfc075542c92a2c63/Dataset/Dataset.xlsx'}
VERSION = 'c0fa6957c7f88a57af29410cfc075542c92a2c63'


def validate(files):
    """Validate the published source snapshot without transforming it.

    :param files: Original filenames mapped to local paths.
    :raises ValueError: If a file has an invalid source format.
    """
    validate_workbook(files['dataset.xlsx'], {'ID', 'Sequence', 'T1/2(hour)', 'Test Sample', 'In-Vivo/Vitro'})


def acquire(directory: Path, *, force=False, local_files=None, provenance=None):
    """Store unchanged raw pepmsnd artifacts and update the shared receipt.

    :param pathlib.Path directory: Destination ``raw/pepmsnd`` directory.
    :param bool force: Replace existing registered source artifacts.
    :param local_files: Optional mapping of original local artifacts to import.
    :param provenance: Original per-file receipts for imported artifacts.
    :return: Source receipt stored within ``raw/manifest.json``.
    :rtype: dict
    :raises ValueError: If source validation or checksum verification fails.
    :raises OSError: If retrieval or storage fails.
    """
    return acquire_files(directory, source='pepmsnd', files=FILES, version=VERSION,
                         validate=validate, force=force, local_files=local_files, provenance=provenance)
