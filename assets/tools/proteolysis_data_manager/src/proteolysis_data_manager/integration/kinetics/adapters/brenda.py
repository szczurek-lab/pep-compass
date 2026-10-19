"""Stream all EC 3.4 entries into the relational contract.

An entry is a source context, not a claim of an independent experiment. Repeated
references, values or names never establish a common experimental setup.
"""
from __future__ import annotations

import json
import math
import re
import tarfile

import ijson

from proteolysis_data_manager.integration.kinetics.schema import SCHEMAS, origin, stable_id

NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
PARAMETERS = {
    'turnover_number': ('kcat', 's^-1', 1.0),
    'km_value': ('km', 'M', 1e-3),
    'kcat_km_value': ('kcat_km', 'M^-1 s^-1', 1e3),
}
UNIT_REFERENCES = {
    'kcat_km': 'https://www.brenda-enzymes.org/search_result.php?a=305',
    'kcat_and_km': 'https://www.brenda-enzymes.org/datafields.php',
}


def text(value):
    return str(value).strip() if value is not None and str(value).strip() else None


def split_value(raw):
    match = re.fullmatch(r'\s*([^{}]*?)\s*(?:\{(.*)\})?\s*', raw, flags=re.S)
    return (match[1].strip(), text(match[2])) if match else (raw.strip(), None)


def parse_value(raw):
    """Only a reported nonnegative finite point becomes a numerical label."""
    value = raw.strip().replace('−', '-')
    if re.fullmatch(NUMBER, value):
        number = float(value)
        return (number, 'eq') if math.isfinite(number) and number >= 0 else (None, 'unknown')
    plus_minus = re.fullmatch(rf'({NUMBER})\s*(?:±|\+/-)\s*({NUMBER})', value)
    if plus_minus and all(math.isfinite(float(v)) and float(v) >= 0 for v in plus_minus.groups()):
        return float(plus_minus[1]), 'eq'
    for sign, relation in [('<=', 'le'), ('>=', 'ge'), ('≤', 'le'), ('≥', 'ge'), ('<', 'lt'), ('>', 'gt')]:
        if re.fullmatch(re.escape(sign) + rf'\s*{NUMBER}', value):
            return None, relation
    if re.fullmatch(rf'{NUMBER}\s*(?:-|–|to)\s*{NUMBER}', value):
        return None, 'range'
    if re.match(r'^(?:~|≈|about\b|approx)', value, flags=re.I):
        return None, 'approx'
    return None, 'unknown'


def scalar_condition(comment, field):
    """Read explicit single assay values; reject ranges and multiple mentions."""
    if re.search(r'\b(?:optimum|optimal|stability|unstable|storage|stored|denaturation|melting|expression|purification)\b', comment, re.I):
        return None
    if field == 'ph':
        pattern = r'\bpH\s*(?:=|:|of|at)?\s*(\d+(?:\.\d+)?)'
    else:
        pattern = r'(?<![\w.−–-])(\d+(?:\.\d+)?)\s*°\s*C\b'
    matches = list(re.finditer(pattern, comment, flags=re.I))
    if len(matches) != 1:
        return None
    match = matches[0]
    prefix, suffix = comment[max(0, match.start() - 35):match.start()], comment[match.end():match.end() + 25]
    if re.search(r'(?:[<>≤≥~≈]|\b(?:to|and|or|between|below|above|about|optimum|optimal))\s*$', prefix, re.I):
        return None
    if re.match(r'\s*(?:[-–±/]|to\b|and\s+\d|or\s+\d)', suffix, re.I):
        return None
    value = float(match[1])
    if field == 'ph' and not 0 <= value <= 14:
        return None
    return value


def reported_variant(comment):
    # Keep source wording rather than guessing that unannotated enzymes are WT.
    if re.search(r'\b(?:mutant|mutation|variant|wild[- ]type|recombinant)\b|\b[A-Z]\d+[A-Z]\b', comment):
        return comment
    return None


def substrate_type(name):
    if re.search(r'coumarin|nitroanilide|naphthylamide|\b(?:Mca|Dnp|Dpa|AMC)\b', name, re.I):
        return 'synthetic_substrate'
    if re.search(r'\b(?:casein|albumin|hemoglobin|haemoglobin|fibrin|fibrinogen|prothrombin|plasminogen|factor\s+[IVX]+)\b', name, re.I):
        return 'protein'
    if len(re.findall(r'\b(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)\b', name)) >= 2:
        return 'peptide'
    return None


def add_origin(record, locator):
    ref = {'source': 'brenda', 'id': locator}
    if ref not in record['source_record_id']:
        record['source_record_id'].append(ref)


def iter_records(archive_path):
    with tarfile.open(archive_path, 'r:*') as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith('.json')]
        if len(members) != 1:
            raise ValueError('Expected one BRENDA JSON member')
        with archive.extractfile(members[0]) as handle:
            yield from ijson.kvitems(handle, 'data')


def context_windows(sequence, position):
    """Slice an already reported bond; never search for fragments or a position."""
    if not sequence or position is None or not 1 <= position < len(sequence):
        return None, None, None
    left, right = sequence[max(0, position - 4):position], sequence[position:position + 4]
    return left, right, left + right if len(left) == len(right) == 4 else None


def adapt_records(records):
    tables = {name: [] for name in SCHEMAS}
    enzymes, substrates = {}, {}
    audit = {'raw_entries': 0, 'ec_blocks_with_measurements': 0, 'notes': [], 'parameter_counts': {k: 0 for k in PARAMETERS}}
    for ec, record in records:
        if not ec.startswith('3.4.'):
            continue
        proteins, references = record.get('protein') or {}, record.get('reference') or {}
        has_entries = False
        for family, (parameter, unit, factor) in PARAMETERS.items():
            for index, entry in enumerate(record.get(family) or []):
                if not isinstance(entry, dict):
                    raise ValueError(f'Unexpected non-object kinetic entry at {ec}/{family}/{index}')
                has_entries = True
                audit['raw_entries'] += 1
                audit['parameter_counts'][family] += 1
                locator = f'data/{ec}/{family}/{index}'
                exp_id, measurement_id = stable_id('experiment', locator), stable_id('measurement', locator)
                raw_value = str(entry.get('value') or '')
                value_text, substrate_name = split_value(raw_value)
                value, relation = parse_value(value_text)
                comment = str(entry.get('comment') or '')
                variant = reported_variant(comment)
                entry_proteins = sorted(set(str(p) for p in entry.get('proteins') or []))
                enzyme_ids = []
                for protein_id in entry_proteins:
                    protein = proteins.get(protein_id) or {}
                    enzyme_id = stable_id('enzyme', ec, protein_id, variant)
                    enzyme_ids.append(enzyme_id)
                    if enzyme_id not in enzymes:
                        enzymes[enzyme_id] = {
                            'enzyme_id': enzyme_id, **origin('brenda', f'data/{ec}/protein/{protein_id}'),
                            'name_reported': text(record.get('recommended_name')), 'ec_number': ec,
                            'uniprot_accession': sorted(set(str(a) for a in protein.get('accessions') or [])) or None,
                            'enzyme_species': text(protein.get('organism')),
                            'enzyme_variant_reported': variant,
                            'description_raw': text(protein.get('comment')),
                        }
                    # Variant-bearing entries retain the actual supporting locator.
                    if variant:
                        add_origin(enzymes[enzyme_id], locator)
                    tables['experiment_enzymes_relation'].append({
                        'experiment_id': exp_id, 'enzyme_id': enzyme_id,
                        'enzyme_link_status': 'direct' if len(entry_proteins) == 1 else 'reported_group_unresolved',
                    })
                substrate_id = None
                if substrate_name:
                    explicit_sequence = text(entry.get('substrate_sequence'))
                    chemistry = {field: text(entry.get(field)) for field in [
                        'chirality', 'topology', 'n_terminal_modification', 'c_terminal_modification', 'other_modifications']}
                    substrate_id = stable_id('substrate', ec, substrate_name, explicit_sequence, chemistry)
                    if substrate_id not in substrates:
                        substrates[substrate_id] = {
                            'substrate_id': substrate_id, **origin('brenda', locator),
                            'name_reported': substrate_name,
                            'substrate_type': text(entry.get('substrate_type')) or substrate_type(substrate_name),
                            'sequence': explicit_sequence, 'sequence_length': len(explicit_sequence) if explicit_sequence else None,
                            **chemistry,
                        }
                    else:
                        add_origin(substrates[substrate_id], locator)
                    tables['experiment_substrates_relation'].append({'experiment_id': exp_id, 'substrate_id': substrate_id})
                publication_ids, publication_urls = [], []
                for ref_id in sorted(set(str(r) for r in entry.get('references') or [])):
                    ref = references.get(ref_id) or {}
                    pmid = text(ref.get('pmid'))
                    publication_ids.append(f'PMID:{pmid}' if pmid else f'brenda:{ec}:reference:{ref_id}')
                    if pmid and pmid.isdigit():
                        publication_urls.append(f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/')
                buffer = re.search(r'[^.;]*\bbuffer\b[^.;]*', comment, re.I)
                ph, temperature = scalar_condition(comment, 'ph'), scalar_condition(comment, 'temperature_c')
                tables['experiments'].append({
                    'experiment_id': exp_id, **origin('brenda', locator),
                    'publication_id': sorted(set(publication_ids)) or None,
                    'publication_urls': sorted(set(publication_urls)) or None,
                    'assay_method_raw': None, 'in_vivo_in_vitro': None,
                    # An enzyme's organism and source tissue do not identify the assay matrix.
                    'species': None, 'matrix': None, 'matrix_fraction': None,
                    'temperature_c': temperature, 'ph': ph,
                    'buffer_raw': buffer[0].strip() if buffer else None, 'conditions_raw': text(comment),
                })
                tables['kinetic_measurements'].append({
                    'measurement_id': measurement_id, 'experiment_id': exp_id, 'substrate_id': substrate_id,
                    'parameter_type': parameter, 'value_numeric': value * factor if value is not None else None,
                    'value_unit': unit, 'value_relation': relation, 'measurement_scope': 'unknown',
                    'sd': None, 'ci_lower': None, 'ci_upper': None, 'ci_level': None, 'replicate_count': None,
                })
                # BRENDA's kinetics families do not define a structured cleavage-site
                # field. Free-text reaction descriptions stay raw until reviewed;
                # no positions are inferred from substrate names or products.
                reasons = []
                if relation != 'eq': reasons.append(f'non-point {relation}; value_numeric=None; original={raw_value!r}')
                if '±' in value_text or '+/-' in value_text: reasons.append('untyped plus/minus retained raw; SD/CI remain null')
                if len(entry_proteins) != 1: reasons.append(f'{len(entry_proteins)} protein links; no independent measurements created')
                if re.search(r'\bpH\b', comment, re.I) and ph is None: reasons.append('pH mention is ambiguous or not an assay scalar; retained raw')
                if '°' in comment and temperature is None: reasons.append('temperature mention is ambiguous or not an assay scalar; retained raw')
                if temperature is not None and (temperature > 150 or temperature < -20):
                    reasons.append(f'reported temperature {temperature:g}°C requires source review; preserved without correction')
                if reasons:
                    audit['notes'].append({'locator': locator, 'measurement_id': measurement_id, 'reasons': reasons})
        audit['ec_blocks_with_measurements'] += int(has_entries)
    tables['enzymes'] = list(enzymes.values())
    tables['substrates'] = list(substrates.values())
    for name in ['enzymes', 'substrates']:
        for row in tables[name]:
            row['source_record_id'].sort(key=lambda r: (r['source'], r['id']))
    return tables, audit


def load(archive_path):
    return adapt_records(iter_records(archive_path))
