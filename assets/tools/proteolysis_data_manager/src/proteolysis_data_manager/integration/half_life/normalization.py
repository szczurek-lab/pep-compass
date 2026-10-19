"""Normalize half-life observations without erasing chemistry or censoring.

Features describe an unmodified reference backbone, never the full chemistry of
modified peptides. Repeated source rows are retained; integration marks shared
representations rather than averaging away experimental conditions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from proteolysis_data_manager.utils.storage import download_manifest
from proteolysis_data_manager.integration.half_life.adapters.common import text, normalized_sequence, known, chemistry_field, medium, measurement, features, row, excel_rows, chemical_signature
from proteolysis_data_manager.integration.half_life.adapters.peplife2 import peplife_row
from proteolysis_data_manager.integration.half_life.adapters.dramp import dramp_rows

from proteolysis_data_manager.integration.half_life.schema import SCHEMA_VERSION


def load_source_records(root: Path, config_dir: Path | None = None) -> pd.DataFrame:
    """Load validated source files and retain all published benchmark views."""
    root = Path(root)
    from proteolysis_data_manager.acquisition.half_life import verify_raw_sources
    verify_raw_sources(root)
    if config_dir is None:
        config_dir = Path(__file__).parent / 'adapters/resources'
    raw_root = root / 'raw'
    from proteolysis_data_manager.integration.half_life.adapters import peplife2, pepmsnd, plifepred, dramp, cavaco
    rows = peplife2.load(raw_root)
    peplife_by_id = {r['source_record_id']: r for r in rows}
    rows.extend(pepmsnd.load(raw_root))
    rows.extend(plifepred.load(raw_root, peplife_by_id))
    rows.extend(dramp.load(raw_root))
    rows.extend(cavaco.load(raw_root, config_dir))
    frame = pd.DataFrame(rows)
    # DRAMP does not have a dedicated in-vitro column. Recover context only
    # when the same publication and recorded chemistry have matching PEPlife
    # observations with one consistent context. A sequence-only match is never
    # enough. All candidate IDs stay visible for inspection.
    peplife_index = {}
    for r in rows:
        if r['source'] == 'peplife2':
            peplife_index.setdefault((r['publication_id'], chemical_signature(r)), []).append(r)
    for idx, r in frame[frame.source.eq('dramp')].iterrows():
        candidates = peplife_index.get((r.publication_id, chemical_signature(r)), [])
        candidates = [c for c in candidates if c['species'] == r.species and c['matrix'] == r.matrix]
        contexts = {c['in_vivo_in_vitro'] for c in candidates}
        if len(contexts) == 1 and 'unknown' not in contexts:
            frame.at[idx, 'in_vivo_in_vitro'] = next(iter(contexts))
            frame.at[idx, 'metadata_origin'] = 'peplife2_publication_and_chemistry_match'
            frame.at[idx, 'candidate_peplife2_ids'] = ';'.join(c['source_record_id'] for c in candidates)
    frame['linked_peplife2_id'] = frame.get('linked_peplife2_id', pd.Series(dtype=str)).fillna('')
    frame['source_locator'] = frame.get('source_locator', pd.Series(dtype=str)).fillna('source record id / workbook row')
    # Attach actual source checksums/version to every observation.
    for source, indices in frame.groupby('source').groups.items():
        manifest = json.loads(download_manifest(root,source,legacy_fallback=True).read_text())
        for idx in indices:
            entry = manifest['files'][frame.at[idx, 'source_filename']]
            frame.at[idx, 'source_file_sha256'] = entry['sha256']
            frame.at[idx, 'source_version'] = manifest['version']
    frame['chemical_identity_complete'] = frame.apply(lambda r: r['is_standard_sequence'] and not r['sequence_contains_lowercase'] and all(known(r[k]) for k in ('chirality', 'topology', 'n_terminal_modification', 'c_terminal_modification', 'chemical_modifications')), axis=1)
    frame['chemical_identity_key'] = frame.apply(lambda r: hashlib.sha256(json.dumps(chemical_signature(r)).encode()).hexdigest()[:20] if r['chemical_identity_complete'] else '', axis=1)
    frame['reference_backbone_matches_chemistry'] = frame['chemical_identity_complete'] & frame['chirality'].eq('l') & frame['topology'].eq('linear') & frame[['n_terminal_modification', 'c_terminal_modification', 'chemical_modifications']].eq('free').all(axis=1)
    frame['is_human_serum'] = frame['species'].eq('human') & frame['matrix'].eq('serum')
    frame['is_exact_positive_half_life'] = frame['measurement_kind'].eq('half_life') & frame['relation'].eq('eq') & frame['half_life_seconds'].gt(0)
    frame['baseline_eligible'] = frame['is_human_serum'] & frame['in_vivo_in_vitro'].eq('in vitro') & frame['is_exact_positive_half_life'] & frame['is_standard_sequence'] & frame['linked_label_status'].isin(['not_linked', 'same_label'])
    from proteolysis_data_manager.integration.half_life.labels import add_duration_classes
    frame = add_duration_classes(frame)
    frame['k_obs_per_second'] = frame['half_life_seconds'].where(frame['is_exact_positive_half_life']).rdiv(math.log(2))
    # A duplicate representation is not another independent observation. PEPlife
    # API rows can differ only in biological activity metadata while reporting
    # the same stability result; authors' natural/mixed views also share rows.
    frame['representation_key'] = frame.apply(_representation_key, axis=1)
    peplife_representations = frame[frame.source.eq('peplife2')].set_index('source_record_id')
    for idx, r in frame[frame.linked_peplife2_id.ne('')].iterrows():
        candidate = peplife_representations.loc[r.linked_peplife2_id]
        if r.relation == candidate.relation and r.relation == 'eq' and math.isclose(r.half_life_seconds, candidate.half_life_seconds, rel_tol=1e-6):
            frame.at[idx, 'representation_key'] = candidate.representation_key
    frame['is_redundant_representation'] = frame.duplicated('representation_key')
    frame['exclusion_reasons'] = frame.apply(_exclusions, axis=1)
    return frame




def _representation_key(r) -> str:
    if r['source'] == 'plifepred':
        payload = ['plifepred', r['source_record_id'], r['half_life_raw']]
    elif r['source'] == 'peplife2':
        # Do not collapse poorly identified records. Published source IDs remain
        # accessible even when fully matching measurements are represented once.
        if not r['chemical_identity_complete']:
            return r['observation_id']
        raw = json.loads(r['raw_record_json'])
        payload = {k: v for k, v in raw.items() if k not in {'id', 'nature', 'activity_sub'}}
    else:
        return r['observation_id']
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _exclusions(r) -> str:
    reasons = []
    if not r['is_human_serum']: reasons.append('not_confirmed_human_serum')
    if r['in_vivo_in_vitro'] != 'in vitro': reasons.append('not_confirmed_in_vitro')
    if not r['is_exact_positive_half_life']: reasons.append('not_exact_positive_half_life')
    if not r['is_standard_sequence']: reasons.append('nonstandard_or_unresolved_sequence')
    if r['linked_label_status'] not in {'not_linked', 'same_label'}: reasons.append(r['linked_label_status'])
    if r['is_redundant_representation']: reasons.append('redundant_source_representation')
    return ';'.join(reasons)


def load_observations(root, config_dir=None):
    """Return the compact, consolidated public contract."""
    from proteolysis_data_manager.integration.half_life.schema import normalize_inventory
    from proteolysis_data_manager.protocols.half_life.conditions import condition_frame
    return normalize_inventory(condition_frame(load_source_records(root, config_dir)))[0]


def build_normalized_dataset(root):
    """Build the canonical snapshot, concrete source remarks and one manifest."""
    from proteolysis_data_manager.acquisition.half_life import SOURCES
    from proteolysis_data_manager.utils.io import sha256, write_json
    from proteolysis_data_manager.protocols.half_life.conditions import condition_frame
    from proteolysis_data_manager.integration.half_life.schema import CLEAN_COLUMNS, normalize_inventory, write_parquet
    from proteolysis_data_manager.integration.half_life.remarks import write_remarks
    root = Path(root)
    inventory = condition_frame(load_source_records(root))
    frame, audit = normalize_inventory(inventory)
    path = root / 'normalized/observations.parquet'
    write_parquet(frame, path)
    write_remarks(inventory, audit, root)
    package = Path(__file__).resolve().parents[2]
    manifest = {
        'schema_version': SCHEMA_VERSION, 'columns': CLEAN_COLUMNS,
        'raw_representations': len(inventory), 'rows': len(frame),
        'raw_source_rows': {source: int(inventory.source.eq(source).sum()) for source in SOURCES},
        'shared_source_measurements': int(frame.source.map(len).gt(1).sum()),
        'enriched_source_records': int(audit.enriched_fields.map(bool).sum()),
        'normalized_parquet_sha256': sha256(path),
        'sources': {source: sha256(download_manifest(root, source, legacy_fallback=True)) for source in SOURCES},
        'implementation_sha256': {str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob('*.py'))},
        'cavaco_curated_table_sha256': sha256(Path(__file__).parent / 'adapters/resources/cavaco_validation.tsv'),
        'class_scheme': 'short <30 min; medium [30 min,6 h]; long >6 h',
        'scope_limitations': [
            'Cavaco: 16 experimental validation peptides; training peptides are not reconstructed.',
            'Reported ranges/censoring/classes are not numerical half-life points.',
            'Documented CI may supply a representative midpoint only if no point is reported; noted in remarks.',
            'Missing CI/SD stay null; unspecified plus/minus errors are documented in source remarks.',
            'No physiological defaults are written to normalized reported pH/temperature fields.',
            'Known conflicts stay separate; sequence coincidence alone never establishes measurement identity.',
            'Normalization consolidates verified aliases; it applies no condition acceptance gates.',
        ],
    }
    write_json(root / 'manifests/normalization_manifest.json', manifest)
    return manifest
