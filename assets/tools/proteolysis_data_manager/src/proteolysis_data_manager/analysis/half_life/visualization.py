"""Readable time axes, stable duration colors and target-environment emphasis."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

DURATION_ORDER = ['unresolved', 'short', 'medium', 'long']
DURATION_COLORS = {'unresolved':'#999999', 'short':'#4477AA', 'medium':'#EE7733', 'long':'#228833'}
DURATION_LABELS = {'unresolved':'unresolved (no unique interval)', 'short':'short (<30 min)',
                   'medium':'medium (30 min–6 h)', 'long':'long (>6 h)'}


def time_label(seconds, _position=None):
    if seconds <= 0: return ''
    if seconds < 60: return f'{seconds:g} s'
    if seconds < 3600: return f'{seconds/60:g} min'
    if seconds < 86400: return f'{seconds/3600:g} h'
    return f'{seconds/86400:g} d'


def readable_time_axis(ax, *, axis='x', scale='log'):
    """Use physical-time labels on either axis and expose the requested scale.

    :param ax: Matplotlib axes containing positive time observations.
    :param str axis: ``x`` or ``y``.
    :param str scale: ``log`` or ``linear``.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    if axis not in {'x', 'y'} or scale not in {'log', 'linear'}:
        raise ValueError('Unsupported time axis or scale')
    getattr(ax, f'set_{axis}scale')(scale)
    target = ax.xaxis if axis == 'x' else ax.yaxis
    lo, hi = getattr(ax, f'get_{axis}lim')()
    if scale == 'log':
        candidates = np.array([.001, .01, .1, 1, 5, 10, 30, 60, 120, 300, 600, 900,
                               1800, 3600, 7200, 21600, 43200, 86400, 604800,
                               2592000, 31536000, 315360000], dtype=float)  # (K,)
        visible = candidates[(candidates >= lo) & (candidates <= hi)]  # (K_visible,)
        if len(visible) > 9:
            keep = set(visible[::int(np.ceil(len(visible) / 7))]) | {1800., 21600.}
            visible = np.array([x for x in visible if x in keep])
        target.set_major_locator(FixedLocator(visible))
        target.set_minor_formatter(NullFormatter())
    target.set_major_formatter(FuncFormatter(time_label))
    if axis == 'x':
        ax.tick_params(axis='x', rotation=35)


def time_bins(values, bins=20):
    values=np.asarray(values, dtype=float)
    values=values[np.isfinite(values) & (values>0)]
    if not len(values): return np.geomspace(1,86400,bins+1)
    lo,hi=values.min(),values.max()
    if lo==hi: lo,hi=lo*.8,hi*1.2
    return np.geomspace(lo,hi,bins+1)


def formatted_table(frame, limit=20):
    """Emphasize target rows and class names without contrasting row fills."""
    preview=frame.head(limit)
    rendered=preview.to_html(index=False,classes='hl-table',na_rep='None',float_format=lambda x:f'{x:.4g}')
    target=(preview.species.isin(['human','mouse']) & preview.matrix.isin(['serum','plasma'])).tolist() if {'species','matrix'} <= set(preview.columns) else [False]*len(preview)
    index=iter(target)
    def row(match):
        focus=next(index,False)
        return '<tr'+(' class="hl-focus"' if focus else '')+'>'+match[1]+'</tr>'
    rendered=re.sub(r'<tr>(.*?)</tr>',row,rendered,flags=re.S)
    for label,color in DURATION_COLORS.items():
        rendered=rendered.replace('<td>'+label+'</td>', '<td><span style="color:'+color+';font-weight:600">'+label+'</span></td>')
    style='<style>.hl-table td,.hl-table th{padding:6px 10px;text-align:left;background:transparent!important;border-bottom:1px solid rgba(128,128,128,.15)}.hl-table{border-collapse:collapse}.hl-focus td{font-weight:650}.hl-focus td:first-child{border-left:3px solid #4477AA}</style>'
    return style+rendered


def duration_summary(frame):
    frame = visualization_frame(frame)
    assigned=frame.duration_class.fillna('unresolved')
    points=frame.regression_label_available if 'regression_label_available' in frame else frame.label_type.eq('exact_point')
    rows=[]
    for label in DURATION_ORDER:
        g=frame[assigned.eq(label)]
        rows.append({'duration_class':label,'time_interval':DURATION_LABELS[label].split('(',1)[1].rstrip(')'),
                     'all_assigned_records':len(g),'exact_point_records':int(points.loc[g.index].sum()),
                     'bounded_or_non_numeric_records':int((~points.loc[g.index]).sum())})
    return pd.DataFrame(rows)


def plot_duration_distributions(frame):
    import matplotlib.pyplot as plt
    frame = visualization_frame(frame)
    fig,axes=plt.subplots(1,3,figsize=(17,4.7))
    frame = visualization_frame(frame)
    assigned=frame.duration_class.fillna('unresolved')
    bins=np.histogram_bin_edges(frame.length,bins=25)
    groups=[frame.loc[assigned.eq(c),'length'] for c in DURATION_ORDER]
    axes[0].hist(groups,bins=bins,stacked=True,color=[DURATION_COLORS[c] for c in DURATION_ORDER],
                 label=[DURATION_LABELS[c] for c in DURATION_ORDER])
    axes[0].set(xlabel='Peptide length (residues)',ylabel='All source records')
    points=frame.regression_label_available if 'regression_label_available' in frame else frame.label_type.eq('exact_point')
    exact=frame[points]
    bins=time_bins(exact.half_life_seconds,bins=30)
    boundaries=[x for x in [1800.,21600.] if bins[0]<x<bins[-1]]
    bins=np.unique(np.r_[bins,boundaries])
    for label in ['short','medium','long']:
        v=exact.loc[exact.duration_class.eq(label),'half_life_seconds']
        axes[1].hist(v,bins=bins,color=DURATION_COLORS[label],edgecolor='white',label=DURATION_LABELS[label])
    for threshold in [1800,21600]: axes[1].axvline(threshold,ls=':',color='#666666',lw=1)
    readable_time_axis(axes[1]); axes[1].set(xlabel='Half-life (physical time; logarithmic spacing)',ylabel='Exact point records')
    summary=duration_summary(frame)
    x=np.arange(len(summary))
    axes[2].bar(x,summary.all_assigned_records,color=[DURATION_COLORS[c] for c in DURATION_ORDER])
    axes[2].set_xticks(x,[DURATION_LABELS[c] for c in DURATION_ORDER],rotation=35,ha='right')
    for i,n in enumerate(summary.all_assigned_records): axes[2].text(i,n,f'{n}',ha='center',va='bottom')
    axes[2].set(ylabel='All source records',title='Exact and class-only records together')
    axes[0].legend(fontsize=8); fig.tight_layout()
    return fig


def visualization_frame(frame):
    """Accept either compact observations or transient rich overview records."""
    if 'half_life_class' not in frame:
        return frame
    result = frame.rename(columns={'half_life_class': 'duration_class', 'sequence_length': 'length'}).copy()
    result['regression_label_available'] = result.half_life_seconds.notna()
    return result


def plot_acceptance_conditions(rows, config):
    """One plot per gate, separating documented values from reference assumptions."""
    import matplotlib.pyplot as plt
    figures = {}
    for field, xlabel in [('ph', 'pH'), ('temperature_c', 'Temperature (°C)')]:
        reported = rows.loc[rows[field].notna(), field + '_effective'].to_numpy(dtype=float)
        assumed = rows.loc[rows[field].isna(), field + '_effective'].to_numpy(dtype=float)
        lo, hi = config['acceptance_bounds'][field]
        reference = config['defaults'][field]
        values = list(reported) + [lo, hi, reference]
        span = max(max(values) - min(values), .2 if field == 'ph' else 2)
        bins = np.linspace(min(values) - span * .1, max(values) + span * .1, 20)
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.hist([reported, assumed], bins=bins, stacked=True, color=['#4477AA', '#BBBBBB'],
                label=[f'Reported (n={len(reported)})', f'Assumed reference (n={len(assumed)})'])
        ax.axvspan(lo, hi, color='#228833', alpha=.12, label=f'Acceptance: {lo:g}–{hi:g}')
        ax.axvline(reference, ls='--', color='#CC6677', lw=2, label=f'Reference: {reference:g}')
        outside = rows[field + '_effective'].notna() & ~rows[field + '_effective'].between(lo, hi)
        ax.set(xlabel=xlabel, ylabel='Selected observations', title=f'{xlabel}: {int(outside.sum())} outside this gate')
        ax.legend(fontsize=8); fig.tight_layout()
        figures[field] = fig
    return figures


def plot_before_after(before, after):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, field, xlabel in [(axes[0], 'sequence_length', 'Peptide length (residues)'),
                              (axes[1], 'half_life_seconds', 'Half-life (physical time; numerical labels)')]:
        a = before[field].dropna().to_numpy(dtype=float)
        b = after[field].dropna().to_numpy(dtype=float)
        bins = time_bins(a, 20) if field == 'half_life_seconds' else np.histogram_bin_edges(a, bins=20)
        ax.hist(a, bins=bins, histtype='step', lw=2, color='#4477AA', label=f'Before (n={len(a)})')
        ax.hist(b, bins=bins, histtype='step', lw=2, color='#EE7733', label=f'Accepted (n={len(b)})')
        if field == 'half_life_seconds':
            readable_time_axis(ax)
            for threshold in [1800, 21600]:
                ax.axvline(threshold, ls=':', color='#999999', lw=1)
        ax.set(xlabel=xlabel, ylabel='Observations'); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# Processed two-table dataset figures reuse the existing time and class styling.
SOURCE_COLORS = {'peplife2': '#4477AA', 'pepmsnd': '#EE6677', 'plifepred': '#228833',
                 'dramp': '#CCBB44', 'cavaco': '#AA3377'}


def plot_processed_missingness(table, *, width=17):
    """Plot percent missing with one denominator per source.

    :param pandas.DataFrame table: Long missingness summary with percentages.
    :param float width: Figure width in inches.
    :return: A labelled heatmap of source metadata missingness.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    pivot = table.pivot(index='source', columns='field', values='missing_percent')
    fig, ax = plt.subplots(figsize=(width, 4.5))
    heat = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=100, cmap='YlOrRd', aspect='auto')  # (S, F)
    counts = table.groupby('source').observations.first()
    ax.set_yticks(range(len(pivot)), [f'{s} (n={counts[s]})' for s in pivot.index])
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=60, ha='right')
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f'{value:.0f}', ha='center', va='center', fontsize=8,
                    color='white' if value > 65 else 'black')
    ax.set_title('Missing metadata (%) — denominator: source observations')
    fig.colorbar(heat, ax=ax, label='Missing (%)')
    fig.tight_layout()
    return fig


def plot_processed_sources(counts, sequence_overlap, peptide_overlap):
    """Show source shares and directional pairwise peptide overlap percentages.

    :param pandas.DataFrame counts: Source observation and peptide counts.
    :param pandas.DataFrame sequence_overlap: Pairwise exact sequence-text counts.
    :param pandas.DataFrame peptide_overlap: Pairwise exact description counts.
    :return: Count bars and labelled overlap matrices.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    source_shares = counts.set_index('source')[
        ['observations_share_percent', 'reported_peptides_union_coverage_percent']
    ].rename(columns={
        'observations_share_percent': 'Observations (% of all observations)',
        'reported_peptides_union_coverage_percent': 'Peptide descriptions (% of union)',
    })
    source_shares.plot.bar(ax=axes[0], color=['#4477AA', '#66CCEE'])
    axes[0].set(ylabel='Share (%)', title='Source shares; peptide coverage may overlap', xlabel='Source')
    axes[0].set_ylim(0, max(100, source_shares.max().max() * 1.15))
    axes[0].tick_params(axis='x', rotation=35)
    overlap_tables = [
        (sequence_overlap, 'sequence_strings', 'Shared sequence strings'),
        (peptide_overlap, 'reported_peptides', 'Shared reported peptide descriptions'),
    ]
    for ax, (table, denominator_column, title) in zip(axes[1:], overlap_tables):
        pivot = table.pivot(index='source_a', columns='source_b', values='shared_count')
        denominator = pivot.index.map(counts.set_index('source')[denominator_column])
        percentages = pivot.div(np.asarray(denominator, dtype=float), axis=0) * 100  # (S, S)
        heat = ax.imshow(percentages.to_numpy(dtype=float), cmap='Blues', vmin=0, vmax=100, aspect='auto')  # (S, S)
        ax.set_xticks(range(len(pivot)), pivot.columns, rotation=35, ha='right')
        ax.set_yticks(range(len(pivot)), pivot.index)
        for i in range(len(pivot)):
            for j in range(len(pivot)):
                percentage = percentages.iloc[i, j]
                count = pivot.iloc[i, j]
                ax.text(j, i, f'{percentage:.1f}%\n(n={count})', ha='center', va='center', fontsize=8,
                        color='white' if percentage > 60 else 'black')
        ax.set_title(f'{title}\n% of row source; n shown')
        fig.colorbar(heat, ax=ax, label='Overlap (%)')
    fig.tight_layout()
    return fig


def plot_processed_conditions(table, *, focus_species=('human', 'mouse')):
    """Count every reported core context and emphasize the requested species.

    :param pandas.DataFrame table: Counts by species, matrix and assay mode.
    :param tuple focus_species: Species with bold labels.
    :return: Horizontal counts, including explicitly missing contexts.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    ordered = table.sort_values('observations', ascending=True)
    fig, ax = plt.subplots(figsize=(13, max(5, .29 * len(ordered))))
    colors = ['#4477AA' if s in focus_species else '#BBBBBB' for s in ordered.species]
    bars = ax.barh(range(len(ordered)), ordered.observations, color=colors)
    ax.set_yticks(range(len(ordered)), ordered.condition_label)
    for tick, species in zip(ax.get_yticklabels(), ordered.species):
        if species in focus_species:
            tick.set_fontweight('bold')
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set(xlabel='Source observations', title='Reported species | matrix | in vivo/in vitro')
    ax.margins(x=.12)
    fig.tight_layout()
    return fig


def plot_processed_agreement(shared, *, groups_per_figure=8, scale='log'):
    """Plot all cross-source points and reported CI for each peptide/context.

    :param pandas.DataFrame shared: Matched observations from shared_measurements.
    :param int groups_per_figure: Number of peptide/context groups per page.
    :param str scale: Physical half-life axis scale, ``linear`` or ``log``.
    :return: All figure pages; an explanatory page if no matches exist.
    :rtype: list[matplotlib.figure.Figure]
    """
    import matplotlib.pyplot as plt
    if groups_per_figure < 1 or scale not in {'log', 'linear'}:
        raise ValueError('Invalid plot pagination or scale')
    if shared.empty:
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.text(.5, .5, 'No shared peptide descriptions with equal reported conditions', ha='center', va='center')
        ax.set_axis_off()
        return [fig]
    groups = list(shared.groupby('comparison_id', sort=True))
    sources = sorted(shared.source.unique())
    offsets = dict(zip(sources, np.linspace(-.3, .3, len(sources))))
    figures = []
    for start in range(0, len(groups), groups_per_figure):
        page = groups[start:start + groups_per_figure]
        fig, ax = plt.subplots(figsize=(max(12, 1.8 * len(page)), 7))
        labels = []
        for x, (comparison_id, rows) in enumerate(page):
            first = rows.iloc[0]
            from textwrap import fill
            numeric = ['matrix_fraction', 'temperature_c', 'ph', 'peptide_concentration_um']
            missing = int(first[numeric].isna().sum())
            context = f'{first.species} {first.matrix}, {first.in_vivo_in_vitro}'
            labels.append(f'{comparison_id}\n{fill(first.sequence, width=20)}\n{context}\nNumeric conditions missing: {missing}/4')
            for source, values in rows.groupby('source'):
                positions = x + offsets[source] + np.linspace(-.035, .035, len(values))  # (N,)
                color = SOURCE_COLORS.get(source, '#666666')
                ax.scatter(positions, values.half_life_seconds.astype(float), color=color, s=25, alpha=.8,
                           label=source if x == 0 else None)
                ci = values.ci_lower_seconds.notna() & values.ci_upper_seconds.notna()
                ax.vlines(positions[ci.to_numpy(dtype=bool)], values.loc[ci, 'ci_lower_seconds'].astype(float),
                          values.loc[ci, 'ci_upper_seconds'].astype(float), color=color, alpha=.6)
        ax.set_xticks(range(len(page)), labels, rotation=45, ha='right', fontsize=8)
        readable_time_axis(ax, axis='y', scale=scale)
        ax.set(ylabel=f'Reported half-life ({scale} scale)',
               title='Human/mouse cross-source records — equal reported peptide and conditions\nMissing conditions are not evidence of experimental equivalence')
        # Keep every source in the legend, including sources absent from page group 1.
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker='o', linestyle='', color=SOURCE_COLORS.get(s, '#666666'), label=s)
                           for s in sorted({s for _, g in page for s in g.source})], fontsize=8)
        ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        figures.append(fig)
    return figures


def plot_processed_durations(values, *, source=None, conditions_per_figure=16, scale='log', focus_species=None):
    """Show condition-specific duration distributions and stability classes.

    :param pandas.DataFrame values: Exact positive half-life observations.
    :param str source: Optional source identifier; None pools source rows.
    :param int conditions_per_figure: Number of contexts on each figure page.
    :param str scale: Time axis scale, ``linear`` or ``log``.
    :param tuple focus_species: Optional species filter applied before pagination.
    :return: Every context page, retaining all observations and their multiplicity.
    :rtype: list[matplotlib.figure.Figure]
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if conditions_per_figure < 1 or scale not in {'log', 'linear'}:
        raise ValueError('Invalid plot pagination or scale')
    rows = values.loc[values.source.eq(source)] if source is not None else values
    if focus_species is not None:
        rows = rows.loc[rows.species.isin(focus_species)]
    title = source if source is not None else 'All sources (source representations retained)'
    groups = sorted(rows.groupby('condition_label'), key=lambda item: (-len(item[1]), item[0]))
    if not groups:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(.5, .5, f'{title}: no exact positive half-lives', ha='center', va='center')
        ax.set_axis_off()
        return [fig]
    figures = []
    all_values = rows.half_life_seconds.to_numpy(dtype=float)  # (N,)
    lo, hi = min(all_values.min() * .7, 900), max(all_values.max() * 1.3, 43200)
    for start in range(0, len(groups), conditions_per_figure):
        page = groups[start:start + conditions_per_figure]
        fig, ax = plt.subplots(figsize=(15, max(4, .55 * len(page))))
        for left, right, label in [(lo, 1800, 'short'), (1800, 21600, 'medium'), (21600, hi, 'long')]:
            ax.axvspan(left, right, color=DURATION_COLORS[label], alpha=.09)
        for y, (condition, group) in enumerate(page):
            durations = group.half_life_seconds.to_numpy(dtype=float)  # (N_g,)
            # Fit the density in displayed coordinates; all points remain visible.
            if len(np.unique(durations)) > 1 and len(durations) >= 3:
                coordinate = np.log10(durations) if scale == 'log' else durations  # (N_g,)
                violin = ax.violinplot([coordinate], positions=[y], vert=False, widths=.65,
                                      showextrema=False, showmeans=False)
                if scale == 'log':
                    for body in violin['bodies']:
                        for path in body.get_paths():
                            path.vertices[:, 0] = 10 ** path.vertices[:, 0]
                for body in violin['bodies']:
                    body.set(facecolor='#777777', edgecolor='#555555', alpha=.2)
            spread = np.linspace(-.18, .18, len(group))  # (N_g,)
            ax.scatter(durations, y + spread, s=10, alpha=.65,
                       c=[DURATION_COLORS[c] for c in group.stability_class])
        ax.set_yticks(range(len(page)), [f'{condition} (n={len(group)})' for condition, group in page])
        for tick, (condition, _) in zip(ax.get_yticklabels(), page):
            if condition.startswith(('human |', 'mouse |')):
                tick.set_fontweight('bold')
        readable_time_axis(ax, axis='x', scale=scale)
        ax.set(xlim=(lo, hi), xlabel=f'Exact half-life ({scale} scale)',
               title=f'{title} — reported condition distributions; page {start // conditions_per_figure + 1}')
        ax.legend(handles=[Line2D([], [], marker='o', linestyle='', color=DURATION_COLORS[c], label=DURATION_LABELS[c])
                           for c in ['short', 'medium', 'long']], loc='upper right', fontsize=8)
        ax.grid(axis='x', alpha=.2)
        fig.tight_layout()
        figures.append(fig)
    return figures


def plot_reported_intervals(table, *, estimates=False, scale='log'):
    """Show reported CI and optionally held-out estimates for every peptide.

    :param pandas.DataFrame table: Reported intervals and optional outer-fold estimates.
    :param bool estimates: Overlay estimated endpoint columns in a second color.
    :param str scale: ``log`` or ``linear`` physical-time y scale.
    :return: Figure with per-peptide sequence, n and concentration labels.
    :rtype: matplotlib.figure.Figure
    :raises ValueError: If the axis scale is invalid or no reported intervals exist.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if scale not in {'log', 'linear'} or table.empty:
        raise ValueError('Reported intervals and a log/linear scale are required')
    rows = table.sort_values(['half_life_seconds', 'experiment_id']).reset_index(drop=True)
    x = np.arange(len(rows), dtype=float)  # (N,)
    fig, ax = plt.subplots(figsize=(max(13, .95 * len(rows)), 7))
    reported_x = x - .12 if estimates else x
    ax.vlines(reported_x, rows.ci_lower_seconds.astype(float), rows.ci_upper_seconds.astype(float), color='#AA3377', linewidth=2)
    ax.scatter(reported_x, rows.half_life_seconds.astype(float), color='#AA3377', s=24, zorder=3)
    ax.hlines(rows.ci_lower_seconds.astype(float), reported_x - .06, reported_x + .06, color='#AA3377')
    ax.hlines(rows.ci_upper_seconds.astype(float), reported_x - .06, reported_x + .06, color='#AA3377')
    handles = [Line2D([], [], color='#AA3377', marker='o', label='Reported point and CI')]
    if estimates:
        estimated_x = x + .12
        ax.vlines(estimated_x, rows.estimated_lower_seconds.astype(float), rows.estimated_upper_seconds.astype(float), color='#0077BB', linewidth=2)
        ax.scatter(estimated_x, rows.half_life_seconds.astype(float), color='#0077BB', marker='x', s=24, zorder=3)
        handles.append(Line2D([], [], color='#0077BB', marker='x', label='Leave-one-peptide-out interval estimate'))
    labels = []
    for row in rows.itertuples():
        n = 'not reported' if pd.isna(row.replicate_count) else f'{row.replicate_count:g}'
        concentration = 'not reported' if pd.isna(row.peptide_concentration_um) else f'{row.peptide_concentration_um:g} µM'
        labels.append(f'{row.source}:{row.source_record_id}\n{row.sequence}\nn={n}; {concentration}')
    ax.set_xticks(x, labels, rotation=65, ha='right', fontsize=8)
    readable_time_axis(ax, axis='y', scale=scale)
    level = ', '.join(f'{100 * level:g}%' for level in sorted(rows.ci_level.dropna().unique()))
    first = rows.iloc[0]
    condition = f'{first.species} {first.matrix}; fraction={first.matrix_fraction:g}; T={first.temperature_c:g} °C'
    title = f'Reported {level} intervals — {condition}'
    if estimates:
        title += '\nEndpoint reconstruction on held-out peptides; statistical coverage is not validated'
    ax.set(ylabel=f'Half-life and interval endpoints ({scale} scale)', title=title)
    ax.legend(handles=handles, fontsize=8)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    return fig
