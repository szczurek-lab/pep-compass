"""Normalize half-life observations without erasing chemistry or censoring.

Features describe an unmodified reference backbone, never the full chemistry of
modified peptides. Repeated source rows are retained; integration marks shared
representations rather than averaging away experimental conditions.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

SCHEMA_VERSION = '2.0'
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
MISSING = {'', 'n.a.', 'n.a', 'na', 'n/a', 'nan', 'not mentioned', 'not given', 'not available', 'not found'}
UNITS = {'s': 1, 'sec': 1, 'second': 1, 'seconds': 1, 'min': 60, 'minute': 60, 'minutes': 60,
         'h': 3600, 'hr': 3600, 'hrs': 3600, 'hour': 3600, 'hours': 3600, 'd': 86400, 'day': 86400, 'days': 86400}
NUMBER = r'(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'
UNIT = r'(?:seconds?|sec|minutes?|min|hours?|hrs?|days?|[shd])'


def text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    return str(value).strip()


def normalized_sequence(value, *, preserve_case=False) -> str:
    """Remove formatting while optionally retaining residue stereochemistry.

    :param value: Published sequence notation.
    :param bool preserve_case: Retain case-sensitive modified residue notation.
    :return: Sequence without HTML or whitespace.
    :rtype: str
    """
    clean = re.sub(r'\s+', '', html.unescape(re.sub(r'<[^>]+>', '', text(value))))
    return clean if preserve_case else clean.upper()


def known(value) -> bool:
    return text(value).lower() not in MISSING | {'none', 'unknown'}


def chemistry_field(value) -> str:
    value = text(value).lower()
    if value in MISSING:
        return 'unknown'
    if value in {'none', 'free', 'no modification', 'unmodified'}:
        return 'free'
    return re.sub(r'\s+', ' ', value)


def medium(value) -> tuple[str, str]:
    """Use the stated sample, never a protease name or the paper's subject."""
    value = text(value).lower()
    matrix = 'mixed' if 'serum' in value and 'plasma' in value else ('serum' if 'serum' in value else ('plasma' if 'plasma' in value else 'other'))
    species = []
    for name, pattern in [('human', r'human|homo sapiens'), ('mouse', r'\b(mouse|mice|murine)\b'),
                          ('rat', r'\brats?\b'), ('monkey', r'monkey|macaque'),
                          ('bovine', r'bovine|\bcow\b'), ('porcine', r'porcine|\bpig\b'),
                          ('dog', r'\bdogs?\b|canine'), ('rabbit', r'rabbit')]:
        if re.search(pattern, value):
            species.append(name)
    return (species[0] if len(species) == 1 else ('mixed' if species else 'unknown'), matrix)


def measurement(value, unit='', *, prose=False) -> dict:
    """Parse exact, approximate, censored and interval half-lives conservatively.

    A remaining fraction at a stated time is not a measured half-life, even if
    the fraction happens to equal 50%. Multiple time values require review.
    """
    raw = text(value)
    clean = html.unescape(re.sub(r'<\/?(?:sub|sup)>', '', raw)).lower()
    clean = clean.replace(',', '.').replace('≤', '<=').replace('≥', '>=').replace('＞', '>').replace('＜', '<').replace('∼', '~').replace('≈', '~').replace('–', '-').replace('−', '-')
    clean = re.sub(r'(?<=\d)\s*[·]\s*(?=\d)', '.', clean)
    result = {'measurement_kind': 'half_life', 'relation': 'unknown', 'half_life_seconds': None,
              'lower_bound_seconds': None, 'upper_bound_seconds': None, 'remaining_fraction': None,
              'observation_time_seconds': None, 'parse_status': 'unparsed',
              'reported_error_seconds': None, 'uncertainty_kind': 'not_reported',
              'half_life_estimate_type': 'unknown'}
    if prose:
        interval = re.search(r'(?:half[ -]?li(?:fe|ves|ve)|t1/2)\s*(?:of|is|was|are|were|:|=)?\s*(' + NUMBER + r')\s*-\s*(' + NUMBER + r')\s*(' + UNIT + r')\b', clean)
        if interval:
            lo, hi = float(interval[1]) * UNITS[interval[3]], float(interval[2]) * UNITS[interval[3]]
            if 0 < lo <= hi:
                result.update(relation='interval', parse_status='parsed', lower_bound_seconds=lo, upper_bound_seconds=hi)
            return result
    # A reported point estimate with unspecified +/- is usable as a point
    # label. Its error is preserved, never renamed SD/SE or converted to CI.
    pm = re.fullmatch(r'\s*(?P<v>'+NUMBER+r')\s*±\s*(?P<e>'+NUMBER+r')\s*', clean)
    if prose:
        # Require an explicit half-life statement and exactly one +/- time.
        times = list(re.finditer(r'(?P<v>'+NUMBER+r')\s*±\s*(?P<e>'+NUMBER+r')\s*(?P<u>'+UNIT+r')\b', clean))
        pm = times[0] if len(times) == 1 and re.search(r'half[ -]?li(?:fe|ves|ve)|t1/2', clean) else None
        if pm is not None and re.search(r'[<>~]=?\s*$|(?:less than|more than|about|approximately)\s*$', clean[:pm.start()]):
            result['parse_status'] = 'unresolved_censored_or_approximate_error'
            return result  # Never let fallback parsing treat the error as a point.
    if pm is not None:
        factor = UNITS.get(pm['u'] if prose else text(unit).lower())
        if factor:
            value, error = float(pm['v'])*factor, float(pm['e'])*factor
            if math.isfinite(value) and math.isfinite(error) and value > 0:
                result.update(relation='eq', half_life_seconds=value, reported_error_seconds=error,
                              uncertainty_kind='unspecified_plus_minus', half_life_estimate_type='reported_point',
                              parse_status='parsed_with_unspecified_error')
            return result
    # A percentage embedded between a bound and its time unit is not a CI.
    # Keep the raw text but allow the explicitly reported time bound to parse.
    if prose:
        clean = re.sub(r'(?<=\d)\([^)]*%[^)]*\)(?=\s*'+UNIT+r'\b)', '', clean)
    # Prevent a number with time units from becoming an invented half-life.
    fraction = re.search(r'(?P<rel>[<>]=?|~|more than|less than|about)?\s*(?P<f>'+NUMBER+r')\s*%.*?(?:after|within|for)\s*(?:an? incubation time of\s*)?(?P<t>'+NUMBER+r')\s*(?P<u>'+UNIT+r')\b', clean)
    if prose and fraction and not re.search(r'half[ -]?li(?:fe|ves)|t1/2', clean):
        result.update(measurement_kind='remaining_fraction', parse_status='fraction_at_time',
                      remaining_fraction=float(fraction['f']) / 100,
                      fraction_relation={'more than': 'gt', 'less than': 'lt', '>': 'gt', '<': 'lt', '>=': 'ge', '<=': 'le', '~': 'approx', 'about': 'approx'}.get(text(fraction['rel']), 'eq'),
                      observation_time_seconds=float(fraction['t']) * UNITS[fraction['u']])
        return result
    relation = r'(?P<rel><=|>=|<|>|~|less than|more than|greater than|at least|at most|about|approximately)?'
    if prose:
        # Prefer a clause explicitly talking about half-life. Other timepoints
        # in the same sentence (complete degradation) must not be extracted.
        clause = re.search(r'(?:half[ -]?li(?:fe|ves|ve)|t1/2)(?:s)?\s*(?:of|is|was|are|were|:|=)?\s*('+relation+r'\s*(?P<v>'+NUMBER+r')\s*(?P<u>'+UNIT+r')\b)', clean)
        if clause is None:
            match_simple = re.fullmatch(relation+r'\s*(?P<v>'+NUMBER+r')\s*(?P<u>'+UNIT+r')\s*', clean)
            if match_simple is None:
                if re.search(r'half[ -]?li(?:fe|ves|ve)', clean):
                    # A half-life may be documented in a longer sentence such
                    # as "half-live is found to be". Only a single time value
                    # is admissible and no percentage readout may intervene.
                    matches = list(re.finditer(relation+r'\s*(?P<v>'+NUMBER+r')\s*(?P<u>'+UNIT+r')\b', clean))
                    if len(matches) == 1 and '%' not in clean:
                        match_simple = matches[0]
                elif re.search(r'antibacterial|antimicrobial|activity', clean):
                    result['measurement_kind'] = 'functional_activity'
            if match_simple is None:
                return result
            clause = match_simple
        factor = UNITS.get(clause['u'])
        match = clause
    else:
        factor = UNITS.get(text(unit).lower())
        match = re.fullmatch(r'\s*'+relation+r'\s*(?P<v>'+NUMBER+r')\s*', clean)
        interval = re.fullmatch(r'\s*('+NUMBER+r')\s*-\s*('+NUMBER+r')\s*', clean)
        if interval and factor:
            lo, hi = [float(v)*factor for v in interval.groups()]
            if 0 < lo <= hi:
                result.update(relation='interval', parse_status='parsed', lower_bound_seconds=lo, upper_bound_seconds=hi)
            return result
    if match is None or factor is None:
        return result
    seconds = float(match['v']) * factor
    if not math.isfinite(seconds) or seconds <= 0:
        result['parse_status'] = 'nonpositive_or_nonfinite'
        return result
    rel = {'<': 'lt', '>': 'gt', '<=': 'le', '>=': 'ge', 'less than': 'lt', 'more than': 'gt',
           'greater than': 'gt', 'at least': 'ge', 'at most': 'le', '~': 'approx', 'about': 'approx',
           'approximately': 'approx'}.get(text(match['rel']), 'eq')
    result.update(relation=rel, parse_status='parsed')
    if rel == 'eq':
        result['half_life_seconds'] = seconds
        result['half_life_estimate_type'] = 'reported_point'
    elif rel == 'approx':
        # Approximate values do not enter the exact-regression label column.
        result['lower_bound_seconds'] = None
        result['upper_bound_seconds'] = None
        result['approximate_half_life_seconds'] = seconds
    elif rel in {'lt', 'le'}:
        result['upper_bound_seconds'] = seconds
    else:
        result['lower_bound_seconds'] = seconds
    return result


@lru_cache(maxsize=8192)
def features(sequence: str) -> dict:
    """Compute reference-backbone descriptors with the same method for all sources."""
    if not sequence or re.fullmatch(f'[{AMINO_ACIDS}]+', sequence) is None:
        return {}
    protein = ProteinAnalysis(sequence)
    counts = protein.count_amino_acids()
    return {'reference_molecular_weight_da': protein.molecular_weight(), 'reference_pi': protein.isoelectric_point(),
            'reference_charge_ph7': protein.charge_at_pH(7), 'reference_gravy': protein.gravy(),
            'reference_aromaticity': protein.aromaticity(), 'reference_nonpolar_fraction': sum(counts[x] for x in 'AVILMFWPG') / len(sequence),
            **{f'fraction_{aa}': counts[aa] / len(sequence) for aa in AMINO_ACIDS}}


def row(source, view, identifier, raw, sequence, value, unit='', *, prose=False, include_features=False, **metadata) -> dict:
    sequence_clean = normalized_sequence(sequence)
    observation = dict(schema_version=SCHEMA_VERSION, source=source, dataset_view=view,
                       source_record_id=str(identifier), sequence_raw=text(sequence), sequence=sequence_clean,
                       length=len(sequence_clean), half_life_raw=text(value), unit_raw=text(unit),
                       species='unknown', matrix='other', in_vivo_in_vitro='unknown',
                       chirality='unknown', topology='unknown', n_terminal_modification='unknown',
                       c_terminal_modification='unknown', chemical_modifications='unknown',
                       publication_id='', assay='', temperature_c=None, ph=None, matrix_fraction=None,
                       peptide_concentration_raw='', test_sample_raw='', n_replicates=None,
                       ci_lower_seconds=None, ci_upper_seconds=None, ci_confidence_level=None, ci_origin='not_reported',
                       half_life_mean_seconds=None, half_life_sd_seconds=None, half_life_se_seconds=None,
                       measurement_index=1, measurement_clause_raw=text(value), smiles='', metadata_origin='direct',
                       linked_label_status='not_linked',
                       raw_record_json=json.dumps(raw, ensure_ascii=False, default=str), **measurement(value, unit, prose=prose))
    observation.update(metadata)
    observation['observation_id'] = f'{source}:{view}:{identifier}'
    observation['is_standard_sequence'] = bool(re.fullmatch(f'[{AMINO_ACIDS}]+', sequence_clean))
    observation['sequence_contains_lowercase'] = any(c.islower() for c in re.sub(r'<[^>]+>', '', text(sequence)))
    if include_features:
        observation.update(features(sequence_clean))
    return observation


def excel_rows(path: Path, required: set[str]) -> list[dict]:
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        values = workbook.active.values
        columns = next(values)
        missing = required - set(columns)
        if missing:
            raise ValueError(f'{path}: required columns missing: {sorted(missing)}')
        return [dict(zip(columns, values)) for values in values if any(v is not None for v in values)]
    finally:
        workbook.close()


def chemical_signature(record: dict) -> tuple:
    return (normalized_sequence(record.get('sequence_raw', record.get('sequence')), preserve_case=True),
            *(record.get(k, 'unknown') for k in ('chirality', 'topology', 'n_terminal_modification', 'c_terminal_modification', 'chemical_modifications')))


def condition_frame(frame):
    """Parse only explicit single molar concentrations and explicit serum %.

    Mass concentrations and administered doses are not converted using a
    reference-backbone molecular weight. No physiological value is imputed.
    """
    result = frame.copy()
    def concentration(raw):
        raw = str(raw).replace('μ', 'µ')
        m = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(nM|µM|uM|mM|M)(?:\s+after 1:1 mixing)?\s*', raw)
        return float(m[1]) * {'nM': .001, 'µM': 1, 'uM': 1, 'mM': 1000, 'M': 1e6}[m[2]] if m else np.nan
    result['peptide_concentration_um'] = result.peptide_concentration_raw.map(concentration)
    result['concentration_numeric_origin'] = np.where(result.peptide_concentration_um.notna(), 'explicit molar value', 'unavailable or incompatible units')
    def fraction(raw):
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*%\s*(?:aqueous\s+)?(?:(?:human|mouse|murine|rat|bovine|porcine)\s+)?(?:serum|plasma)\b', str(raw), re.I)
        if len(matches) == 1 and 0 <= float(matches[0]) <= 100:
            return float(matches[0]) / 100
        if not matches and re.search(r'\b(?:full|neat|undiluted)\s+(?:(?:human|mouse|murine|rat)\s+)?(?:serum|plasma)\b', str(raw), re.I):
            return 1.0
        return np.nan
    missing = result.matrix_fraction.isna()
    result['matrix_fraction_origin'] = np.where(missing, 'unavailable', 'normalized source metadata')
    parsed = result.test_sample_raw.map(fraction)
    recover = missing & parsed.notna()
    result.loc[recover, 'matrix_fraction'] = parsed[recover]
    result.loc[recover, 'matrix_fraction_origin'] = 'explicit percentage in test_sample_raw'
    # Read experimental-condition fields only; sequence pI is not assay pH.
    text_fields = [f for f in ['incubation_time_raw', 'test_sample_raw', 'measurement_clause_raw'] if f in result]
    def explicit_value(record, pattern):
        values = set()
        if pattern.startswith('(?<!') and record.get('temperature_metadata_scope') == 'detector_preparation_not_half_life':
            return np.nan
        for field in text_fields:
            raw = str(record.get(field, '')).replace('℃', '°C')
            # Multiple values and ranges are deliberately not collapsed.
            range_pattern = (r'\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*°?\s*C\b'
                             if pattern.startswith('(?<!') else r'\bpH\s*[:=]?\s*\d+(?:\.\d+)?\s*[-–]\s*\d')
            if re.search(range_pattern, raw, re.I):
                continue
            values.update(float(m) for m in re.findall(pattern, raw, re.I))
        return next(iter(values)) if len(values) == 1 else np.nan
    for field, pattern in [('temperature_c', r'(?<![\d.])(\d+(?:\.\d+)?)\s*°?\s*C\b'),
                           ('ph', r'\bpH\s*[:=]?\s*(\d+(?:\.\d+)?)\b')]:
        parsed = result.apply(lambda record: explicit_value(record, pattern), axis=1)
        recover = result[field].isna() & parsed.notna()
        result.loc[recover, field] = parsed[recover]
    for field in ('ph', 'temperature_c', 'matrix_fraction', 'peptide_concentration_um'):
        result[field] = pd.to_numeric(result[field], errors='coerce')
    logger.debug("Parsed explicit assay conditions for %s observations", len(result))
    return result
