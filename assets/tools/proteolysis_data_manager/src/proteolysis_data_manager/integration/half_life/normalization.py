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
from proteolysis_data_manager.integration.half_life.adapters.common import text, normalized_sequence, known, chemistry_field, medium, measurement, features, row, excel_rows, chemical_signature
from proteolysis_data_manager.integration.half_life.adapters.peplife2 import peplife_row
from proteolysis_data_manager.integration.half_life.adapters.dramp import dramp_rows

from proteolysis_data_manager.integration.half_life.schema import SCHEMA_VERSION


def load_source_records(root: Path, config_dir: Path | None = None, *, include_features=True) -> pd.DataFrame:
    """Inspect source records through the single maintained source orchestrator.

    :param pathlib.Path root: Data series root.
    :param pathlib.Path config_dir: Optional reviewed Cavaco transcription directory.
    :param bool include_features: Compute legacy overview descriptors in memory.
    :return: Detailed compatibility view for existing scientific overviews.
    :rtype: pandas.DataFrame
    """
    from proteolysis_data_manager.integration.half_life.datasets import load_adapter_records

    frame = pd.DataFrame(load_adapter_records(root, config_dir))
    if include_features:
        descriptors = pd.DataFrame([features(sequence) for sequence in frame.sequence], index=frame.index)
        frame = pd.concat([frame, descriptors], axis=1)
    frame['linked_peplife2_id'] = frame.get('linked_peplife2_id', pd.Series(dtype=str)).fillna('')
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
    """Forward the previous builder name to the maintained two-table workflow.

    :param pathlib.Path root: Data series root.
    :return: The peptide and experiment tables written under ``processed``.
    :rtype: HalfTimeDataset
    """
    from proteolysis_data_manager.integration.half_life.datasets import build_half_time_datasets
    return build_half_time_datasets(root)
