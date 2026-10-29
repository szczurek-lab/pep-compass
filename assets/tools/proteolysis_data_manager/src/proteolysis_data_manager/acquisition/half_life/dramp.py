"""Acquire the published dramp artifacts without adapting their records."""
from pathlib import Path
from proteolysis_data_manager.acquisition.helpers.raw import acquire_files, validate_workbook

FILES = {'stability_amps.xlsx': 'https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/stability_amps.xlsx'}
VERSION = 'unversioned_public_snapshot'


def validate(files):
    """Validate the published source snapshot without transforming it.

    :param files: Original filenames mapped to local paths.
    :raises ValueError: If a file has an invalid source format.
    """
    validate_workbook(files['stability_amps.xlsx'], {'DRAMP_ID', 'Sequence', 'half_life', 'Type'})


def acquire(directory: Path, *, force=False, local_files=None, provenance=None):
    """Store unchanged raw dramp artifacts and update the shared receipt.

    :param pathlib.Path directory: Destination ``raw/dramp`` directory.
    :param bool force: Replace existing registered source artifacts.
    :param local_files: Optional mapping of original local artifacts to import.
    :param provenance: Original per-file receipts for imported artifacts.
    :return: Source receipt stored within ``raw/manifest.json``.
    :rtype: dict
    :raises ValueError: If source validation or checksum verification fails.
    :raises OSError: If retrieval or storage fails.
    """
    return acquire_files(directory, source='dramp', files=FILES, version=VERSION,
                         validate=validate, force=force, local_files=local_files, provenance=provenance)
