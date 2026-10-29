"""Join original MEROPS dictionaries and matrices into a separate CSV view."""
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from proteolysis_data_manager.acquisition.window_neighbourhood import merops
from proteolysis_data_manager.acquisition.window_neighbourhood.download_all import default_root
from proteolysis_data_manager.utils.io import sha256
from proteolysis_data_manager.utils.logger import get_custom_logger
from proteolysis_data_manager.utils.storage import read_download_receipt

logger = get_custom_logger(__name__)
COLUMNS = (
    'dataset_index', 'folder', 'scientific', 'role', 'apex_index', 'label', 'taxids',
    'peptidase_row', 'code', 'name', 'n_cleavages', 'serum_category',
    'position_index', 'position', 'amino_acid_index', 'amino_acid', 'count', 'logprob',
)


def merge_merops(root=None, *, output_path=None):
    """Export every original matrix cell with its dataset and axis identities.

    :param root: Dataset-series root containing the acquired ``raw/merops`` tree.
    :param output_path: Destination CSV; defaults to ``processed/merops_matrices.csv``.
    :return: CSV path, written only after source validation and a complete export.
    :rtype: pathlib.Path
    :raises ValueError: If source checksums, dictionaries or matrix values disagree,
        or if the output is inside ``raw``.
    :raises OSError: If source files are absent or writing fails.

    One row represents one ``(dataset_index, peptidase_row, position_index,
    amino_acid_index)``. Repeated codes across datasets remain separate. Counts
    and log-probabilities are supplied values: no smoothing, pooling, numerical
    normalization or deduplication is performed. Original files are read only.
    """
    root = Path(root) if root is not None else default_root()
    raw = root / 'raw'
    directory = raw / 'merops'
    output = Path(output_path) if output_path is not None else root / 'processed/merops_matrices.csv'
    if output.resolve().is_relative_to(raw.resolve()):
        raise ValueError('The merged MEROPS CSV must be outside raw')

    # Verify the registered snapshot before joining its original dictionaries.
    receipt = read_download_receipt(root, 'merops')
    files = {name: directory / name for name in receipt['files']}
    for name, entry in receipt['files'].items():
        if sha256(files[name]) != entry['sha256']:
            raise ValueError(f'Acquired MEROPS file changed: {name}')
    merops.validate(files)
    index = json.loads((directory / 'index.json').read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Merging original MEROPS matrix cells into %s', output)

    # Join by explicit dataset IDs and matrix rows; keep every dataset membership.
    count = 0
    with tempfile.TemporaryDirectory(prefix='.merops-', dir=output.parent) as temporary:
        candidate = Path(temporary) / 'matrices.csv'
        with candidate.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for key in sorted(index['by_index'], key=int):
                entry = index['by_index'][key]
                folder = directory / entry['folder']
                metadata = json.loads((folder / 'codes.json').read_text())
                counts = np.load(folder / 'matrices_counts.npy', allow_pickle=False)  # (N, P, A)
                logprob = np.load(folder / 'matrices_logprob.npy', allow_pickle=False)  # (N, P, A)
                dataset_fields = [entry['dataset_index'], entry['folder'], entry['scientific'],
                                  entry['role'], entry['apex_index'], entry['label'],
                                  json.dumps(entry['taxids'], ensure_ascii=False, separators=(',', ':'))]
                for enzyme in sorted(metadata['peptidases'], key=lambda item: item['row']):
                    row = enzyme['row']
                    enzyme_fields = [row, enzyme['code'], enzyme['name'], enzyme['n_cleavages'],
                                     enzyme.get('serum_category', '')]
                    for position_index, position in enumerate(metadata['positions']):
                        for amino_acid_index, amino_acid in enumerate(metadata['amino_acids']):
                            writer.writerow([
                                *dataset_fields, *enzyme_fields, position_index, position,
                                amino_acid_index, amino_acid,
                                float(counts[row, position_index, amino_acid_index]),
                                float(logprob[row, position_index, amino_acid_index]),
                            ])
                            count += 1
        candidate.replace(output)
    logger.info('Exported %s original MEROPS matrix cells', count)
    return output
