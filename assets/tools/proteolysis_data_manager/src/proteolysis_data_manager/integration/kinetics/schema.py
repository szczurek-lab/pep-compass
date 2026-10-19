"""Eight typed tables; nullable measurements never erase their source context."""
from __future__ import annotations

import hashlib
import json

import pyarrow as pa

SCHEMA_VERSION = '1.0'
STRING = pa.string()
FLOAT = pa.float64()
INTEGER = pa.int64()
STRINGS = pa.list_(pa.field('element', STRING))
REFERENCES = pa.list_(pa.field('element', pa.struct([('source', STRING), ('id', STRING)])))
ORIGIN = [('source', STRINGS), ('source_record_id', REFERENCES)]


def table_schema(fields):
    return pa.schema(fields, metadata={b'kinetics_schema_version': SCHEMA_VERSION.encode()})


SCHEMAS = {
    'enzymes': table_schema([
        ('enzyme_id', STRING), *ORIGIN, ('name_reported', STRING), ('ec_number', STRING),
        ('uniprot_accession', STRINGS), ('enzyme_species', STRING),
        ('enzyme_variant_reported', STRING), ('description_raw', STRING),
    ]),
    'substrates': table_schema([
        ('substrate_id', STRING), *ORIGIN, ('name_reported', STRING), ('substrate_type', STRING),
        ('sequence', STRING), ('sequence_length', INTEGER), ('chirality', STRING),
        ('topology', STRING), ('n_terminal_modification', STRING),
        ('c_terminal_modification', STRING), ('other_modifications', STRING),
    ]),
    'experiments': table_schema([
        ('experiment_id', STRING), *ORIGIN, ('publication_id', STRINGS),
        ('publication_urls', STRINGS), ('assay_method_raw', STRING), ('in_vivo_in_vitro', STRING),
        ('species', STRING), ('matrix', STRING), ('matrix_fraction', FLOAT),
        ('temperature_c', FLOAT), ('ph', FLOAT), ('buffer_raw', STRING), ('conditions_raw', STRING),
    ]),
    'experiment_enzymes_relation': table_schema([
        ('experiment_id', STRING), ('enzyme_id', STRING), ('enzyme_link_status', STRING),
    ]),
    'experiment_substrates_relation': table_schema([
        ('experiment_id', STRING), ('substrate_id', STRING),
    ]),
    'kinetic_measurements': table_schema([
        ('measurement_id', STRING), ('experiment_id', STRING), ('substrate_id', STRING),
        ('parameter_type', STRING), ('value_numeric', FLOAT), ('value_unit', STRING),
        ('value_relation', STRING), ('measurement_scope', STRING),
        ('sd', FLOAT), ('ci_lower', FLOAT), ('ci_upper', FLOAT),
        ('ci_level', FLOAT), ('replicate_count', INTEGER),
    ]),
    'cleavage_sites': table_schema([
        ('site_id', STRING), ('experiment_id', STRING), ('substrate_id', STRING), ('enzyme_id', STRING),
        ('position_reported', INTEGER), ('coordinate_basis', STRING), ('bond_reported', STRING),
        ('site_text_raw', STRING), ('cleavage_context_p4_to_p1', STRING),
        ('cleavage_context_p1prime_to_p4prime', STRING), ('cleavage_context_8aa', STRING),
        ('context_origin', STRING), ('source_locator', STRING),
    ]),
    'measurement_cleavage_sites_relation': table_schema([
        ('measurement_id', STRING), ('site_id', STRING),
    ]),
}
PRIMARY_KEYS = {
    'enzymes': ['enzyme_id'], 'substrates': ['substrate_id'], 'experiments': ['experiment_id'],
    'kinetic_measurements': ['measurement_id'], 'cleavage_sites': ['site_id'],
    'experiment_enzymes_relation': ['experiment_id', 'enzyme_id'],
    'experiment_substrates_relation': ['experiment_id', 'substrate_id'],
    'measurement_cleavage_sites_relation': ['measurement_id', 'site_id'],
}
PARAMETER_UNITS = {'kcat': 's^-1', 'km': 'M', 'kcat_km': 'M^-1 s^-1'}
VALUE_RELATIONS = {'eq', 'lt', 'le', 'gt', 'ge', 'range', 'approx', 'unknown'}


def stable_id(kind, *parts):
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return f'brenda:{kind}:' + hashlib.sha256(payload.encode()).hexdigest()[:24]


def origin(source, locator):
    return {'source': [source], 'source_record_id': [{'source': source, 'id': locator}]}


def arrow_table(name, rows):
    return pa.Table.from_pylist(rows, schema=SCHEMAS[name])
