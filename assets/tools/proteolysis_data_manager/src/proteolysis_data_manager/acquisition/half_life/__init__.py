"""Raw half-life acquisition; no scientific adapters or unit normalization."""
from pathlib import Path
from proteolysis_data_manager.acquisition.half_life.download_all import SOURCES, acquire, acquire_all
from proteolysis_data_manager.acquisition.half_life.peplife2 import VERSION as PEPLIFE_VERSION, records, validate_response
from proteolysis_data_manager.acquisition.half_life.pepmsnd import VERSION as PEPMSND_COMMIT
from proteolysis_data_manager.acquisition.helpers.raw import validate_container
from proteolysis_data_manager.utils.paths import default_data_root
from proteolysis_data_manager.utils.storage import read_download_receipt
from proteolysis_data_manager.utils.io import sha256


def validate_file(path):
    """Validate a source container, retaining the preceding import interface.

    :param path: Original source file.
    :raises ValueError: If validation fails.
    """
    path = Path(path)
    return validate_response(path) if path.suffix == '.json' else validate_container(path)


def peplife_records(root=None):
    """Return unchanged API rows after FASTA completeness verification.

    :param root: Half-life data root.
    :return: Original PEPlife2 records.
    :rtype: list[dict]
    """
    root = Path(root) if root is not None else default_data_root()
    return records({name: root / 'raw/peplife2' / name for name in SOURCES['peplife2']})


def verify_raw_sources(root=None):
    """Verify acquired source bytes before downstream adapters read them.

    :param root: Half-life data root.
    :raises ValueError: If a source differs from its acquisition receipt.
    """
    root = Path(root) if root is not None else default_data_root()
    for source, files in SOURCES.items():
        receipt = read_download_receipt(root, source)
        for name in files:
            if sha256(root / 'raw' / source / name) != receipt['files'][name]['sha256']:
                raise ValueError(f'Source changed after acquisition: {source}/{name}')
    peplife_records(root)
