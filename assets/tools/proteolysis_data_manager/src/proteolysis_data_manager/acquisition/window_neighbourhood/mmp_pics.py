"""Acquire published MMP/PICS workbooks, WebPICS results and the primary paper."""
from pathlib import Path
from zipfile import ZipFile

from proteolysis_data_manager.acquisition.helpers.raw import acquire_files, validate_container

VERSION = 'PMC4777984.1'
BASE_URL = f'https://pmc-oa-opendata.s3.amazonaws.com/{VERSION}'
FILES = {name: f'{BASE_URL}/{name}' for name in
         (f'{VERSION}.xml', *(f'mmc{i}.xlsx' for i in range(4, 10)), 'mmc10.zip')}
# The primary paper defines the synthetic substrates and assay interpretation.
FILES['Activesite.pdf'] = (
    'https://digibuo.uniovi.es/dspace/bitstream/handle/10651/36861/'
    'Activesite.pdf?sequence=1&isAllowed=y'
)


def validate(files):
    """Check the published article, workbook containers and archive integrity.

    :param files: Published filenames mapped to local paths.
    :raises ValueError: If a source container is invalid.
    """
    for name, path in files.items():
        validate_container(path)
        if name.endswith('.zip'):
            with ZipFile(path) as archive:
                if not archive.namelist() or archive.testzip() is not None:
                    raise ValueError('Invalid original WebPICS result archive')


def acquire(directory: Path, *, force=False, local_files=None, provenance=None):
    """Store original publication artifacts without extracting cleavage windows.

    :param pathlib.Path directory: Destination ``raw/mmp_pics`` directory.
    :param bool force: Replace registered source files.
    :param local_files: Optional original source artifacts to import.
    :param provenance: Original per-file import receipts.
    :return: Source receipt in the shared raw manifest.
    :rtype: dict
    :raises ValueError: If container or checksum verification fails.
    """
    return acquire_files(directory, source='mmp_pics', files=FILES, version=VERSION,
                         validate=validate, force=force, local_files=local_files, provenance=provenance)
