"""Explicit assay-condition selection, preserving missingness and raw units."""
from __future__ import annotations
import json
import re
import numpy as np
import pandas as pd
from proteolysis_data_manager.integration.half_life.adapters.common import known, condition_frame

PARAMETERS = {
    'ph': 'Assay pH',
    'temperature_c': 'Assay temperature (°C)',
    'matrix_fraction': 'Serum fraction (v/v; 1 = neat serum)',
    'peptide_concentration_um': 'Peptide concentration (µM; molar values only)',
}


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
