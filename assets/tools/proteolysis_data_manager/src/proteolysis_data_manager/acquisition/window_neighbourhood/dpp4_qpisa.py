"""Acquire the original published DPP4/qPISA workbooks."""
from pathlib import Path

from proteolysis_data_manager.acquisition.helpers.raw import acquire_files, validate_container

VERSION = 'PMC11612144.1'
BASE_URL = f'https://pmc-oa-opendata.s3.amazonaws.com/{VERSION}'
FILES = {name: f'{BASE_URL}/{name}' for name in
         (f'{VERSION}.xml', *(f'44320_2024_71_MOESM{i}_ESM.xlsx' for i in range(2, 10)))}


def validate(files):
    """Validate original article and workbook containers without data mapping.

    :param files: Published filenames mapped to local paths.
    :raises ValueError: If a source container is invalid.
    """
    for path in files.values():
        validate_container(path)


def acquire(directory: Path, *, force=False, local_files=None, provenance=None):
    """Store publication workbooks with their original names, values and units.

    :param pathlib.Path directory: Destination ``raw/dpp4_qpisa`` directory.
    :param bool force: Replace registered source files.
    :param local_files: Optional original source artifacts to import.
    :param provenance: Original per-file import receipts.
    :return: Source receipt in the shared raw manifest.
    :rtype: dict
    :raises ValueError: If source container or checksum validation fails.
    """
    return acquire_files(directory, source='dpp4_qpisa', files=FILES, version=VERSION,
                         validate=validate, force=force, local_files=local_files, provenance=provenance)
