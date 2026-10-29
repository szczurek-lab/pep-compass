"""Import Karol's existing MEROPS dictionaries and matrices without modification."""
import csv
import json
from pathlib import Path

import numpy as np

from proteolysis_data_manager.acquisition.helpers.raw import acquire_files
from proteolysis_data_manager.utils.logger import get_custom_logger
from proteolysis_data_manager.utils.paths import repository_root
from proteolysis_data_manager.utils.storage import read_download_receipt

VERSION = 'local-curated-snapshot'
logger = get_custom_logger(__name__)


def default_source_root():
    """Return the existing, independently prepared MEROPS source directory.

    :return: Repository directory containing the original ``index.json``.
    :rtype: pathlib.Path
    """
    return repository_root() / 'data/merops'


def validate(files):
    """Check dictionary identities and matrix axes without rebuilding any values.

    :param files: Original relative filenames mapped to candidate local paths.
    :raises ValueError: If dictionary identities, axes or source matrices disagree.
    :raises KeyError: If a required original file or metadata field is missing.
    """
    index = json.loads(Path(files['index.json']).read_text())
    entries = index['by_index']
    logger.debug('Validating %s original MEROPS datasets', len(entries))
    if index['n_datasets'] != len(entries) or len(index['by_folder']) != len(entries):
        raise ValueError('MEROPS index dataset counts disagree')
    positions, amino_acids = index['positions'], index['amino_acids']
    if len(set(positions)) != len(positions) or len(set(amino_acids)) != len(amino_acids):
        raise ValueError('MEROPS matrix axis labels must be unique')
    folders = set()

    # Validate original dataset identities and ordered peptidase dictionaries.
    for key, entry in entries.items():
        folder = entry['folder']
        if (entry['dataset_index'] != int(key) or folder in folders
                or Path(folder).name != folder or folder in {'.', '..'}
                or index['by_folder'].get(folder) != entry):
            raise ValueError('MEROPS index identities or folder dictionaries disagree')
        folders.add(folder)
        metadata = json.loads(Path(files[f'{folder}/codes.json']).read_text())
        peptidases = metadata['peptidases']
        if metadata['positions'] != positions or metadata['amino_acids'] != amino_acids:
            raise ValueError(f'MEROPS matrix axis labels disagree: {folder}')
        rows = [item['row'] for item in peptidases]
        codes = [item['code'] for item in peptidases]
        if (len(peptidases) != entry['n_peptidases']
                or any(type(row) is not int for row in rows)
                or sorted(rows) != list(range(len(rows))) or len(set(codes)) != len(codes)):
            raise ValueError(f'MEROPS peptidase row identities disagree: {folder}')

        # Validate supplied arrays and their original per-code CSV representations.
        expected_shape = (len(peptidases), len(positions), len(amino_acids))
        matrices = {}
        for kind in ('counts', 'logprob'):
            matrix = np.load(files[f'{folder}/matrices_{kind}.npy'], allow_pickle=False)  # (N, P, A)
            if matrix.shape != expected_shape or not np.isfinite(matrix).all():
                raise ValueError(f'MEROPS matrix shape or values disagree: {folder}/{kind}')
            matrices[kind] = matrix
        if (matrices['counts'] < 0).any():
            raise ValueError(f'MEROPS counts cannot be negative: {folder}')
        for item in peptidases:
            with Path(files[f'{folder}/{item["code"]}.csv']).open(newline='') as handle:
                reader = csv.reader(handle)
                if next(reader) != ['position', *amino_acids]:
                    raise ValueError(f'MEROPS CSV amino-acid labels disagree: {folder}/{item["code"]}')
                values = list(reader)
            if [row[0] for row in values] != positions:
                raise ValueError(f'MEROPS CSV position labels disagree: {folder}/{item["code"]}')
            counts = np.asarray([row[1:] for row in values], dtype=np.float64)  # (P, A)
            if not np.array_equal(counts, matrices['counts'][item['row']]):
                raise ValueError(f'MEROPS CSV and NPY counts disagree: {folder}/{item["code"]}')


def acquire(directory: Path, *, force=False, local_files=None, provenance=None, source_root=None):
    """Copy all original source files, including dictionaries, CSV, NPY and PT.

    :param pathlib.Path directory: Destination ``raw/merops`` directory.
    :param bool force: Replace registered imported files.
    :param local_files: Optional complete relative-filename/path mapping to import.
    :param provenance: Optional original per-file retrieval metadata.
    :param source_root: Original directory; defaults to repository ``data/merops``.
    :return: Local-import source receipt in the shared raw manifest.
    :rtype: dict
    :raises ValueError: If source validation or registered checksums disagree.
    :raises FileNotFoundError: If the original dictionaries have not been supplied.
    """
    directory = Path(directory)
    previous = read_download_receipt(directory.parent.parent, 'merops', missing_ok=True)
    if source_root is None and local_files is None and not force and 'index.json' in previous.get('files', {}):
        # A registered snapshot remains usable after moving it away from its import origin.
        return acquire_files(directory, source='merops', version=VERSION, validate=validate,
                             files={name: entry['requested_url'] for name, entry in previous['files'].items()},
                             local_files={name: Path(entry['imported_from'])
                                          for name, entry in previous['files'].items()},
                             provenance=previous['files'])
    if local_files is None:
        source_root = Path(source_root) if source_root is not None else default_source_root()
        if not (source_root / 'index.json').is_file():
            raise FileNotFoundError(f'Original MEROPS dictionaries are missing: {source_root / "index.json"}')
        if directory.resolve().is_relative_to(source_root.resolve()):
            raise ValueError('MEROPS destination must be outside the original source directory')
        local_files = {path.relative_to(source_root).as_posix(): path
                       for path in sorted(source_root.rglob('*')) if path.is_file()}
        logger.info('Importing %s original MEROPS files from %s', len(local_files), source_root)
    files = {name: Path(path).resolve().as_uri() for name, path in local_files.items()}
    return acquire_files(directory, source='merops', files=files, version=VERSION,
                         validate=validate, force=force, local_files=local_files, provenance=provenance)
