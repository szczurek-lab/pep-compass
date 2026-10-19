"""Summaries keep measurement, association and source-context counts distinct."""
import numpy as np
import pandas as pd


def table_inventory(dataset):
    from proteolysis_data_manager.integration.kinetics.schema import SCHEMAS
    return pd.DataFrame([{'table': name, 'rows': len(dataset[name]), 'columns': len(schema)} for name, schema in SCHEMAS.items()])


def missingness(dataset):
    return pd.DataFrame([{'table': name, 'field': field, 'missing': int(frame[field].isna().sum()),
                          'total': len(frame), 'missing_percent': 100 * frame[field].isna().mean() if len(frame) else None}
                         for name, frame in dataset.tables.items() for field in frame.columns])


def species_coverage(dataset):
    pairs = dataset['experiment_enzymes_relation'].merge(dataset['enzymes'][['enzyme_id', 'enzyme_species']], on='enzyme_id', validate='many_to_one')
    pairs['enzyme_species'] = pairs.enzyme_species.fillna('not_reported')
    rows = []
    measurements = dataset['kinetic_measurements']
    for species, group in pairs.groupby('enzyme_species'):
        ids = set(group.experiment_id)
        m = measurements.loc[measurements.experiment_id.isin(ids)]
        rows.append({'enzyme_species': species, 'source_contexts': len(ids), 'measurements': len(m),
                     'numeric': int(m.value_numeric.notna().sum()), 'non_numeric': int(m.value_numeric.isna().sum())})
    unlinked = set(dataset['experiments'].experiment_id) - set(pairs.experiment_id)
    if unlinked:
        m = measurements.loc[measurements.experiment_id.isin(unlinked)]
        rows.append({'enzyme_species': 'no_source_protein_link', 'source_contexts': len(unlinked),
                     'measurements': len(m), 'numeric': int(m.value_numeric.notna().sum()),
                     'non_numeric': int(m.value_numeric.isna().sum())})
    return pd.DataFrame(rows, columns=['enzyme_species','source_contexts','measurements','numeric','non_numeric']).sort_values('measurements', ascending=False)


def parameter_summary(view):
    rows = []
    for (parameter, unit), group in view.groupby(['parameter_type', 'value_unit']):
        values = group.value_numeric.dropna()
        rows.append({'parameter': parameter, 'unit': unit, 'measurements': len(group),
                     'numeric': len(values), 'non_numeric': int(group.value_numeric.isna().sum()),
                     'min': values.min(), 'median': values.median(), 'max': values.max()})
    return pd.DataFrame(rows)


def formatted_table(frame, limit=30):
    html = frame.head(limit).to_html(index=False, na_rep='None', float_format=lambda x: f'{x:.4g}', classes='kinetics-table')
    css = '<style>.kinetics-table td,.kinetics-table th{background:transparent!important;border-bottom:1px solid #8883;padding:6px 10px;text-align:left;max-width:420px;overflow-wrap:anywhere;white-space:normal}.kinetics-table{border-collapse:collapse}</style>'
    return css + html


def plot_parameters(view, comparison=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterSciNotation
    parameters = [('kcat', 's⁻¹'), ('km', 'M'), ('kcat_km', 'M⁻¹ s⁻¹')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (parameter, unit) in zip(axes, parameters):
        values = view.loc[view.parameter_type.eq(parameter) & view.value_numeric.gt(0), 'value_numeric'].to_numpy(dtype=float)
        other = comparison.loc[comparison.parameter_type.eq(parameter) & comparison.value_numeric.gt(0), 'value_numeric'].to_numpy(dtype=float) if comparison is not None else np.array([])
        union = np.r_[values, other]
        if len(union):
            low, high = union.min(), union.max()
            if low == high: low, high = low / 1.2, high * 1.2
            bins = np.geomspace(low, high, 30)
            ax.hist(values, bins=bins, color='#4477AA', alpha=.5, label=f'Before (n={len(values)})' if comparison is not None else f'Reported positive points (n={len(values)})')
            if comparison is not None:
                ax.hist(other, bins=bins, histtype='step', color='#CC6677', linewidth=2, label=f'After (n={len(other)})')
            ax.set_xscale('log'); ax.xaxis.set_major_formatter(LogFormatterSciNotation())
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'No positive numerical points', transform=ax.transAxes, ha='center')
        ax.set(xlabel=f'{parameter} ({unit}; logarithmic spacing)', ylabel='Measurements')
    fig.tight_layout()
    return fig


def plot_conditions(view, references=None, bounds=None):
    import matplotlib.pyplot as plt
    references, bounds = references or {}, bounds or {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, field, label in zip(axes, ['ph', 'temperature_c'], ['Reported assay pH', 'Reported assay temperature (°C)']):
        values = view[field].dropna()
        # Keep unusual reported values visible without allowing one source typo
        # to collapse the biologically relevant part of the temperature axis.
        compressed = field == 'temperature_c' and len(values) and (values.max() > 120 or values.min() < -20)
        if len(values):
            bins = np.unique(np.r_[np.linspace(min(values.min(), 0), max(values.max(), 60), 26), np.arange(0, 62.5, 2.5)]) if compressed else 25
            ax.hist(values, bins=bins, color='#4477AA', alpha=.8)
        if compressed:
            from matplotlib.ticker import FixedLocator, FuncFormatter
            ax.set_xscale('symlog', linthresh=60)
            ticks = [0, 20, 37, 60, 100, 200, 400]
            while ticks[-1] < values.max(): ticks.append(ticks[-1] * 2)
            if values.min() < 0:
                ticks = sorted(set(ticks + [-v for v in ticks if v <= abs(values.min()) * 2]))
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:g}'))
            label += '; spacing compressed beyond 60°C'
        if field in references:
            ax.axvline(references[field], ls='--', color='#CC6677', label=f'Physiological reference: {references[field]:g} (display only)')
        if bounds.get(field) is not None:
            ax.axvspan(*bounds[field], color='#228833', alpha=.15, label='Enabled acceptance interval')
        ax.set(xlabel=label, ylabel='Measurements', title=f'Reported: {len(values)}; missing: {len(view) - len(values)}')
        if field in references or bounds.get(field) is not None: ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
