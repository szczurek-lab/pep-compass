"""Acquire the published cavaco artifacts without adapting their records."""
from pathlib import Path
from proteolysis_data_manager.acquisition.helpers.raw import acquire_files, validate_container

FILES = {'article.xml': 'https://pmc-oa-opendata.s3.amazonaws.com/PMC8301568.1/PMC8301568.1.xml',
 'article.pdf': 'https://pmc-oa-opendata.s3.amazonaws.com/PMC8301568.1/PMC8301568.1.pdf',
 'supplement.docx': 'https://pmc-oa-opendata.s3.amazonaws.com/PMC8301568.1/CTS-14-1349-s001.docx',
 'supplement.pdf': 'https://pmc-oa-opendata.s3.amazonaws.com/PMC8301568.1/CTS-14-1349-s003.pdf'}
VERSION = 'PMC8301568.1'


def validate(files):
    """Validate the published source snapshot without transforming it.

    :param files: Original filenames mapped to local paths.
    :raises ValueError: If a file has an invalid source format.
    """
    for path in files.values():
        validate_container(path)


def acquire(directory: Path, *, force=False, local_files=None, provenance=None):
    """Store unchanged raw cavaco artifacts and update the shared receipt.

    :param pathlib.Path directory: Destination ``raw/cavaco`` directory.
    :param bool force: Replace existing registered source artifacts.
    :param local_files: Optional mapping of original local artifacts to import.
    :param provenance: Original per-file receipts for imported artifacts.
    :return: Source receipt stored within ``raw/manifest.json``.
    :rtype: dict
    :raises ValueError: If source validation or checksum verification fails.
    :raises OSError: If retrieval or storage fails.
    """
    return acquire_files(directory, source='cavaco', files=FILES, version=VERSION,
                         validate=validate, force=force, local_files=local_files, provenance=provenance)
