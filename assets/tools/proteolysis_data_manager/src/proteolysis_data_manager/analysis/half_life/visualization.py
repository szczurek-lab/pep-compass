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


def readable_time_axis(ax):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.set_xscale('log')
    lo,hi=ax.get_xlim()
    # Fixed physical-time ticks: no arithmetic conversion from log10 values.
    candidates=np.array([.001,.01,.1,1,5,10,30,60,300,900,1800,3600,21600,86400,
                         604800,2592000,31536000,315360000], dtype=float)
    visible=candidates[(candidates>=lo) & (candidates<=hi)]
    if len(visible)>10:
        # Preserve classification boundaries when thinning crowded axes.
        keep=set(visible[::int(np.ceil(len(visible)/8))]) | {1800.,21600.}
        visible=np.array([x for x in visible if x in keep])
    ax.xaxis.set_major_locator(FixedLocator(visible))
    ax.xaxis.set_major_formatter(FuncFormatter(time_label))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='x',rotation=35)


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
