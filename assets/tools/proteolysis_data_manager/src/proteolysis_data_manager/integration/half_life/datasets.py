"""Single source orchestration and TSV I/O for the two half-time datasets."""
from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd

from proteolysis_data_manager.integration.half_life.adapters import peplife2, pepmsnd, plifepred, dramp, cavaco
from proteolysis_data_manager.integration.half_life.adapters.common import chemical_signature
from proteolysis_data_manager.integration.half_life.adapters.schema import (
    HalfTimeDataset, SEQUENCE_COLUMNS, EXPERIMENT_COLUMNS, FLOAT_COLUMNS,
    adapt_records, typed_frame,
)
from proteolysis_data_manager.utils.logger import get_custom_logger
from proteolysis_data_manager.utils.paths import default_data_root
from proteolysis_data_manager.utils.storage import read_download_receipt

logger = get_custom_logger(__name__)
TABLES = {'sequences_half_time': SEQUENCE_COLUMNS, 'half_time_experiments': EXPERIMENT_COLUMNS}


def load_adapter_records(root, config_dir=None):
    """Read each acquired source once and recover only verified source links.

    :param pathlib.Path root: Data series root with raw files and receipt.
    :param pathlib.Path config_dir: Optional reviewed Cavaco transcription directory.
    :return: Detailed source records for common contract adaptation or inspection.
    :rtype: list[dict]
    :raises ValueError: If raw checksums, source formats or reviewed values differ.
    """
    from proteolysis_data_manager.acquisition.half_life import verify_raw_sources

    root = Path(root)
    logger.info('Verifying raw half-life inputs in %s', root / 'raw')
    verify_raw_sources(root)
    raw_root = root / 'raw'
    config_dir = Path(config_dir) if config_dir is not None else Path(__file__).parent / 'adapters/resources'

    # Preserve existing source parsers and author-ID links.
    rows = peplife2.load(raw_root)
    peplife_by_id = {r['source_record_id']: r for r in rows}
    for module in (pepmsnd, plifepred, dramp, cavaco):
        source = module.__name__.rsplit('.', 1)[-1]
        logger.info('Reading %s', source)
        if source == 'plifepred':
            records = module.load(raw_root, peplife_by_id)
        elif source == 'cavaco':
            records = module.load(raw_root, config_dir)
        else:
            records = module.load(raw_root)
        rows.extend(records)

    # Reuse the reviewed publication + chemistry + sample matching rule once.
    index = {}
    for r in peplife_by_id.values():
        if r['publication_id']:
            key = (r['publication_id'], chemical_signature(r), r['species'], r['matrix'])
            index.setdefault(key, []).append(r)
    for r in rows:
        if r['source'] != 'dramp':
            continue
        key = (r['publication_id'], chemical_signature(r), r['species'], r['matrix'])
        candidates = index.get(key, [])
        contexts = {c['in_vivo_in_vitro'] for c in candidates}
        if len(contexts) == 1 and 'unknown' not in contexts:
            r['in_vivo_in_vitro'] = next(iter(contexts))
            r['metadata_origin'] = 'peplife2_publication_and_chemistry_match'
            r['candidate_peplife2_ids'] = ';'.join(c['source_record_id'] for c in candidates)

    # Attach source receipt fields without reconstructing or modifying raw data.
    receipts = {source: read_download_receipt(root, source) for source in {r['source'] for r in rows}}
    for r in rows:
        receipt = receipts[r['source']]
        r['source_file_sha256'] = receipt['files'][r['source_filename']]['sha256']
        r['source_version'] = receipt['version']
        r['source_locator'] = r.get('source_locator') or f"{r['source_filename']}#{r['source_record_id']}"
    logger.info('Loaded %s source observations', len(rows))
    return rows


def build_half_time_datasets(root=None, *, output_dir=None, config_dir=None):
    """Build and stage both human-readable half-time TSV datasets.

    :param pathlib.Path root: Data series root; defaults to the repository series.
    :param pathlib.Path output_dir: Destination; defaults to ``root/processed``.
    :param pathlib.Path config_dir: Optional reviewed Cavaco transcription directory.
    :return: Validated tables written as UTF-8 TSV, with empty fields for nulls.
    :rtype: HalfTimeDataset
    :raises ValueError: If raw or output contract validation fails.
    :raises OSError: If files cannot be read or written.
    """
    root = Path(root) if root is not None else default_data_root()
    destination = Path(output_dir) if output_dir is not None else root / 'processed'
    dataset = adapt_records(load_adapter_records(root, config_dir))

    # Stage the complete pair before replacing any previous output.
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.half-time-', dir=destination) as temporary:
        for name in TABLES:
            getattr(dataset, name).to_csv(Path(temporary) / f'{name}.tsv', sep='\t', index=False, na_rep='')
        for name in TABLES:
            (Path(temporary) / f'{name}.tsv').replace(destination / f'{name}.tsv')
            logger.info('Wrote %s', destination / f'{name}.tsv')
    return dataset


def read_half_time_datasets(directory):
    """Read both TSV tables, retaining string IDs and explicit missing values.

    :param pathlib.Path directory: Directory containing the processed TSV pair.
    :return: Validated, typed peptide and experiment tables.
    :rtype: HalfTimeDataset
    :raises ValueError: If columns, numeric values or linked keys are invalid.
    :raises OSError: If an output file is missing or unreadable.
    """
    frames = {}
    for name, columns in TABLES.items():
        strings = {c: 'string' for c in columns if c not in FLOAT_COLUMNS | {'replicate_count'}}
        frame = pd.read_csv(Path(directory) / f'{name}.tsv', sep='\t', dtype=strings,
                            keep_default_na=False, na_values=[''], float_precision='round_trip')
        if list(frame.columns) != columns:
            raise ValueError(f'Unexpected {name} columns')
        frames[name] = typed_frame(frame.to_dict('records'), columns)
    return HalfTimeDataset(**frames).validate()
