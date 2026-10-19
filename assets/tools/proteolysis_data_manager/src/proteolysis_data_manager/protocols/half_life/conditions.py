"""Explicit assay-condition selection, preserving missingness and raw units."""
from __future__ import annotations
import json
import re
import numpy as np
import pandas as pd
from proteolysis_data_manager.integration.half_life.normalization import known

PARAMETERS = {
    'ph': 'Assay pH',
    'temperature_c': 'Assay temperature (°C)',
    'matrix_fraction': 'Serum fraction (v/v; 1 = neat serum)',
    'peptide_concentration_um': 'Peptide concentration (µM; molar values only)',
}


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
    for field in PARAMETERS:
        result[field] = pd.to_numeric(result[field], errors='coerce')
    return result


def counts(rows):
    return {'records': len(rows), 'sequences': rows.sequence.nunique(),
            'documented_molecules': rows.loc[rows.chemical_identity_complete, 'chemical_identity_key'].nunique(),
            'publications': rows.loc[rows.publication_id.map(known), 'publication_id'].nunique()}


def group_mask(frame, key):
    mask = pd.Series(True, index=frame.index)
    for field, expected in key.items():
        mask &= frame[field].eq(expected)
    return mask


def condition_status(frame, bounds):
    """Three states: a known mismatch overrides any missing conditions."""
    rejected = pd.Series(False, index=frame.index)
    unknown = rejected.copy()
    for field, interval in bounds.items():
        if interval is None:
            continue
        lo, hi = interval
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError(f'Invalid interval for {field}: {interval}')
        values = pd.to_numeric(frame[field], errors='coerce')
        unknown |= values.isna()
        rejected |= values.notna() & ~values.between(lo, hi)
    return pd.Series(np.where(rejected, 'outside', np.where(unknown, 'unknown', 'accepted')), index=frame.index)


def selection_tables(frame, key, bounds):
    selected = frame[group_mask(frame, key)].copy()
    rows = [{'criterion': 'Selected group', **counts(selected)}]
    eligible = selected[selected.baseline_eligible & ~selected.is_redundant_representation].copy()
    rows.append({'criterion': 'Exact label / supported sequence / consistent label / integrated representative', **counts(eligible)})
    cumulative = eligible
    audit = []
    for field, interval in bounds.items():
        if interval is None:
            continue
        status = condition_status(eligible, {field: interval})
        for state in ['accepted', 'outside', 'unknown']:
            audit.append({'parameter': field, 'status': state, **counts(eligible[status.eq(state)])})
        cumulative = cumulative[condition_status(cumulative, {field: interval}).eq('accepted')]
        rows.append({'criterion': f'{field} within {interval}; documented value required', **counts(cumulative)})
    eligible['condition_status'] = condition_status(eligible, bounds)
    return eligible, pd.DataFrame(rows), pd.DataFrame(audit)


def selection_key(key, bounds):
    return json.dumps({'group': key, 'bounds': bounds, 'bounds_inclusive': True,
                       'missing_policy': 'unknown; never accepted by strict selection',
                       'label_policy': 'baseline_eligible', 'representation_policy': 'integrated'}, sort_keys=True)


def plot_conditions(frame, bounds, references, *, groupby=None):
    """One figure per parameter; shared bins and limits across sources.

    Each subplot includes the full denominator, missing count and acceptance
    counts. An empty distribution still displays reference lines and bounds.
    """
    import matplotlib.pyplot as plt
    groups = list(frame.groupby(groupby, sort=True)) if groupby else [('Selected human-serum group', frame)]
    figures = {}
    for field, label in PARAMETERS.items():
        values = frame[field].dropna()
        interval = bounds.get(field)
        reference = references.get(field, {}).get('value')
        landmarks = list(values)
        if interval is not None: landmarks += list(interval)
        if reference is not None: landmarks += [reference]
        lo, hi = (min(landmarks), max(landmarks)) if landmarks else (0, 1)
        pad = max((hi-lo)*.08, .05 if field in {'ph', 'matrix_fraction'} else 1)
        bins = np.linspace(lo-pad, hi+pad, 18)
        fig, axes = plt.subplots(len(groups), 2, figsize=(12, max(3.5, 2.6*len(groups))), squeeze=False,
                                 gridspec_kw={'width_ratios': [3, 1.4]})
        for (name, rows), (ax, summary) in zip(groups, axes):
            observed = rows[field].dropna()
            ax.hist(observed, bins=bins, color='#4477AA', edgecolor='white', label=f'Documented values (n={len(observed)})')
            if interval is not None:
                ax.axvspan(*interval, color='#228833', alpha=.15, label='Proposed acceptance interval')
                for limit in interval: ax.axvline(limit, color='#228833', ls=':', lw=1)
            if reference is not None:
                ax.axvline(reference, color='#CC6677', ls='--', lw=2, label=references[field]['label'])
            if observed.empty:
                ax.text(.5, .5, 'No documented numeric values', transform=ax.transAxes, ha='center')
            ax.set(xlim=(bins[0], bins[-1]), xlabel=label, ylabel='Records', title=f'{name} | total n={len(rows)}; missing n={len(rows)-len(observed)}')
            ax.legend(fontsize=8, loc='upper right')
            status = condition_status(rows, {field: interval}) if interval is not None else pd.Series(np.where(rows[field].notna(), 'documented', 'unknown'), index=rows.index)
            categories = ['accepted', 'outside', 'unknown'] if interval is not None else ['documented', 'unknown']
            totals = [int(status.eq(x).sum()) for x in categories]
            summary.barh(categories, totals, color=['#228833', '#EE7733', '#BBBBBB'] if interval is not None else ['#4477AA', '#BBBBBB'])
            for i, n in enumerate(totals): summary.text(n, i, f' {n}', va='center')
            summary.set(xlabel='Records', xlim=(0, max(totals+[1])*1.2))
        fig.suptitle(label + ' — measured values, reference and proposed decision', fontsize=14)
        fig.tight_layout()
        figures[field] = fig
    return figures
