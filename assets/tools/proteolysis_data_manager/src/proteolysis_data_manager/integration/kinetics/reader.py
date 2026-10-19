"""Read, validate and subset a relational dataset without multiplying results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from proteolysis_data_manager.utils.io import sha256, write_json
from proteolysis_data_manager.integration.kinetics.schema import (
    SCHEMAS, SCHEMA_VERSION, PRIMARY_KEYS, PARAMETER_UNITS, VALUE_RELATIONS,
)


class KineticsDataset:
    """All contract tables, including typed empty cleavage tables."""

    def __init__(self, tables, metadata=None):
        self.tables = tables
        self.metadata = metadata or {}

    def __getitem__(self, name):
        return self.tables[name]

    @classmethod
    def read(cls, folder, *, manifest_path=None, verify_checksums=True):
        folder = Path(folder)
        if manifest_path is None:
            if folder.name != 'normalized':
                raise ValueError('Pass the centralized manifest_path for a processed bundle')
            manifest_path = folder.parent / 'manifests/normalization_brenda.json'
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported kinetics schema version')
        tables = {}
        for name, schema in SCHEMAS.items():
            path = folder / f'{name}.parquet'
            if verify_checksums and sha256(path) != manifest['tables'][name]['sha256']:
                raise ValueError(f'Changed kinetics table: {name}')
            arrow = pq.read_table(path)
            if not arrow.schema.equals(schema, check_metadata=True):
                raise ValueError(f'Incorrect table schema: {name}')
            tables[name] = arrow.to_pandas()
        result = cls(tables, manifest)
        result.validate()
        return result

    def validate(self):
        if set(self.tables) != set(SCHEMAS):
            raise ValueError('The complete eight-table kinetics bundle is required')
        for name, schema in SCHEMAS.items():
            frame = self[name]
            if list(frame.columns) != schema.names:
                raise ValueError(f'Unexpected columns/order: {name}')
            keys = PRIMARY_KEYS[name]
            if frame[keys].isna().any().any() or frame.duplicated(keys).any():
                raise ValueError(f'Missing or repeated primary key: {name}')
            for field in schema:
                if pa.types.is_floating(field.type) or pa.types.is_integer(field.type):
                    values = frame[field.name].dropna()
                    if not np.isfinite(values).all():
                        raise ValueError(f'Non-finite numeric field: {name}.{field.name}')
                    if pa.types.is_integer(field.type) and (values % 1 != 0).any():
                        raise ValueError(f'Non-integral field: {name}.{field.name}')
            if 'source' in frame:
                for sources, references in zip(frame.source, frame.source_record_id):
                    sources, references = list(sources), list(references)
                    if not sources or sources != sorted(set(sources)):
                        raise ValueError(f'Invalid source list: {name}')
                    if ({r['source'] for r in references} != set(sources) or
                            any(set(r) != {'source', 'id'} or not r['id'] for r in references)):
                        raise ValueError(f'Invalid source references: {name}')
        def foreign(table, column, target, *, nullable=False):
            values = self[table][column]
            if (not nullable and values.isna().any()) or not values.dropna().isin(self[target][column]).all():
                raise ValueError(f'Dangling foreign key: {table}.{column}')
        for name in ['experiment_enzymes_relation', 'experiment_substrates_relation', 'kinetic_measurements', 'cleavage_sites']:
            foreign(name, 'experiment_id', 'experiments')
        foreign('experiment_enzymes_relation', 'enzyme_id', 'enzymes')
        foreign('experiment_substrates_relation', 'substrate_id', 'substrates')
        foreign('kinetic_measurements', 'substrate_id', 'substrates', nullable=True)
        foreign('cleavage_sites', 'substrate_id', 'substrates')
        foreign('cleavage_sites', 'enzyme_id', 'enzymes', nullable=True)
        foreign('measurement_cleavage_sites_relation', 'measurement_id', 'kinetic_measurements')
        foreign('measurement_cleavage_sites_relation', 'site_id', 'cleavage_sites')
        substrate_pairs = set(map(tuple, self['experiment_substrates_relation'][['experiment_id', 'substrate_id']].to_numpy()))
        enzyme_pairs = set(map(tuple, self['experiment_enzymes_relation'][['experiment_id', 'enzyme_id']].to_numpy()))
        for table in ['kinetic_measurements', 'cleavage_sites']:
            for row in self[table].itertuples():
                if pd.notna(row.substrate_id) and (row.experiment_id, row.substrate_id) not in substrate_pairs:
                    raise ValueError(f'Substrate outside experiment: {table}')
                if table == 'cleavage_sites' and pd.notna(row.enzyme_id) and (row.experiment_id, row.enzyme_id) not in enzyme_pairs:
                    raise ValueError('Cleavage enzyme outside experiment')
        measurements = self['kinetic_measurements'].set_index('measurement_id')
        sites = self['cleavage_sites'].set_index('site_id')
        for link in self['measurement_cleavage_sites_relation'].itertuples():
            m, s = measurements.loc[link.measurement_id], sites.loc[link.site_id]
            if m.experiment_id != s.experiment_id or pd.isna(m.substrate_id) or m.substrate_id != s.substrate_id:
                raise ValueError('Measurement/site context mismatch')
            if m.measurement_scope not in {'site_specific', 'site_set'}:
                raise ValueError('A whole-substrate/unknown measurement cannot be assigned to sites')
        m = self['kinetic_measurements']
        if not m.measurement_scope.isin(['unknown', 'substrate_total', 'site_specific', 'site_set']).all():
            raise ValueError('Unknown measurement scope')
        if not self['experiment_enzymes_relation'].enzyme_link_status.isin(['direct', 'reported_group_unresolved', 'reported_complex']).all():
            raise ValueError('Unknown enzyme link status')
        if not m.value_relation.isin(VALUE_RELATIONS).all():
            raise ValueError('Unknown value relation')
        if (m.value_numeric.notna() & m.value_relation.ne('eq')).any():
            raise ValueError('Only reported points populate value_numeric')
        for parameter, unit in PARAMETER_UNITS.items():
            if not m.loc[m.parameter_type.eq(parameter), 'value_unit'].eq(unit).all():
                raise ValueError(f'Unexpected normalized unit: {parameter}')
        if not m.parameter_type.isin(PARAMETER_UNITS).all():
            raise ValueError('Unsupported parameter type')
        if (m.ci_lower.isna() != m.ci_upper.isna()).any() or (m.ci_level.notna() & m.ci_lower.isna()).any():
            raise ValueError('Incomplete reported CI')
        if (m.ci_lower > m.ci_upper).any() or (m.ci_level.notna() & ~m.ci_level.between(0, 1, inclusive='neither')).any():
            raise ValueError('Invalid CI')
        if (m.sd.dropna() < 0).any() or (m.replicate_count.dropna() < 1).any():
            raise ValueError('Invalid uncertainty/replicate count')
        exp = self['experiments']
        if (exp.matrix_fraction.notna() & ~exp.matrix_fraction.between(0, 1)).any():
            raise ValueError('Invalid matrix fraction')
        if (exp.ph.notna() & ~exp.ph.between(0, 14)).any():
            raise ValueError('Invalid pH')
        for row in self['cleavage_sites'].itertuples():
            left, right, whole = row.cleavage_context_p4_to_p1, row.cleavage_context_p1prime_to_p4prime, row.cleavage_context_8aa
            if pd.notna(whole) and (not isinstance(left, str) or not isinstance(right, str) or len(left) != 4 or len(right) != 4 or whole != left + right):
                raise ValueError('Invalid eight-residue context')
            if pd.notna(row.position_reported) and row.position_reported < 1:
                raise ValueError('Invalid reported cleavage position')
        return self

    def subset_experiments(self, experiment_ids):
        ids = set(experiment_ids)
        if not ids <= set(self['experiments'].experiment_id):
            raise ValueError('Unknown experiment identifier')
        tables = {name: frame.copy() for name, frame in self.tables.items()}
        for name in ['experiments', 'experiment_enzymes_relation', 'experiment_substrates_relation', 'kinetic_measurements', 'cleavage_sites']:
            tables[name] = tables[name].loc[tables[name].experiment_id.isin(ids)].copy()
        enzyme_ids = set(tables['experiment_enzymes_relation'].enzyme_id)
        substrate_ids = set(tables['experiment_substrates_relation'].substrate_id)
        tables['enzymes'] = tables['enzymes'].loc[tables['enzymes'].enzyme_id.isin(enzyme_ids)].copy()
        tables['substrates'] = tables['substrates'].loc[tables['substrates'].substrate_id.isin(substrate_ids)].copy()
        links = tables['measurement_cleavage_sites_relation']
        tables['measurement_cleavage_sites_relation'] = links.loc[
            links.measurement_id.isin(tables['kinetic_measurements'].measurement_id) &
            links.site_id.isin(tables['cleavage_sites'].site_id)].copy()
        return KineticsDataset(tables, self.metadata).validate()

    def for_enzyme(self, enzyme_id):
        if enzyme_id not in set(self['enzymes'].enzyme_id):
            raise KeyError(enzyme_id)
        links = self['experiment_enzymes_relation']
        return self.subset_experiments(links.loc[links.enzyme_id.eq(enzyme_id), 'experiment_id'])

    def measurements_view(self):
        """Exactly one row per measurement; enzyme and site associations are lists."""
        result = self['kinetic_measurements'].merge(self['experiments'], on='experiment_id', validate='many_to_one')
        substrate = self['substrates'].drop(columns=['source', 'source_record_id'])
        result = result.merge(substrate, on='substrate_id', how='left', validate='many_to_one')
        links = self['experiment_enzymes_relation']
        enzyme_ids = links.groupby('experiment_id').enzyme_id.agg(list)
        species = links.merge(self['enzymes'][['enzyme_id', 'enzyme_species']], on='enzyme_id', validate='many_to_one')
        species = species.groupby('experiment_id').enzyme_species.agg(lambda v: sorted(set(v.dropna())))
        result['enzyme_ids'] = result.experiment_id.map(enzyme_ids).map(lambda v: v if isinstance(v, list) else [])
        result['enzyme_species'] = result.experiment_id.map(species).map(lambda v: v if isinstance(v, list) else [])
        sites = self['measurement_cleavage_sites_relation'].groupby('measurement_id').site_id.agg(list)
        result['site_ids'] = result.measurement_id.map(sites).map(lambda v: v if isinstance(v, list) else [])
        return result

    def write(self, folder, manifest_path, *, metadata=None):
        self.validate()
        folder, manifest_path = Path(folder), Path(manifest_path)
        folder.mkdir(parents=True, exist_ok=True)
        receipts = {}
        for name, schema in SCHEMAS.items():
            path = folder / f'{name}.parquet'
            pq.write_table(pa.Table.from_pandas(self[name], schema=schema, preserve_index=False).replace_schema_metadata(schema.metadata), path)
            receipts[name] = {'rows': len(self[name]), 'sha256': sha256(path)}
        manifest = {'schema_version': SCHEMA_VERSION, 'tables': receipts, **(metadata or {})}
        write_json(manifest_path, manifest)
        return manifest


def load_kinetics_data(root, *, condition=None, source='brenda'):
    """Load one complete normalized or reviewed source-protocol bundle."""
    if source != 'brenda':
        raise ValueError('Only BRENDA is implemented at this stage')
    root = Path(root)
    if condition is None:
        return KineticsDataset.read(root / 'normalized')
    import re
    if not re.fullmatch(r'[a-z][a-z0-9_]*', condition):
        raise ValueError('Invalid condition key')
    manifest = json.loads((root / 'manifests' / f'protocol_{condition}_{source}.json').read_text())
    if manifest['normalization_manifest_sha256'] != sha256(root / 'manifests/normalization_brenda.json'):
        raise ValueError('Processed bundle uses a stale normalized dataset; rerun the protocol')
    from proteolysis_data_manager.protocols.kinetics import filtering
    if manifest['protocol_sha256'] != sha256(Path(filtering.__file__)):
        raise ValueError('Processed bundle uses a stale protocol; rerun it')
    return KineticsDataset.read(root / 'processed' / condition / 'sources' / source,
                               manifest_path=root / 'manifests' / f'protocol_{condition}_{source}.json')
