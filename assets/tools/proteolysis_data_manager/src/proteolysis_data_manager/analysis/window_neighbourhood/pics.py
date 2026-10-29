"""Read original synthetic-substrate QF measurements from the MMP/PICS deposit."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)
MMP_TO_MEROPS = {
    'MMP1': 'M10.001', 'MMP2': 'M10.003', 'MMP3': 'M10.005', 'MMP7': 'M10.008',
    'MMP8': 'M10.002', 'MMP9': 'M10.004', 'MMP12': 'M10.009', 'MMP13': 'M10.013', 'MMP14': 'M10.014',
}
SUBSTRATE_LABELS = ('PLG↓L', 'PAN↓L', 'PLN↓L', 'PAG↓L', 'VLG↓L', 'VAN↓L')
POSITIONS = ('P4', 'P3', 'P2', 'P1', 'P1p', 'P2p', 'P3p', 'P4p')
PRIMARY_DOI = '10.1016/j.matbio.2015.09.003'


def read_qf_assay(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read QF fit coefficients, reported summaries, raw traces and substrate chemistry.

    :param path: Original ``mmc8.xlsx`` workbook.
    :type path: str | pathlib.Path
    :return: ``rounds``, ``summary``, ``traces``, ``substrates`` and ``round_audit``
        frames, retaining source worksheet and cell/row locations.
    :rtype: dict[str, pandas.DataFrame]
    :raises ValueError: If labels, source layout or numeric values are unsupported.
    :raises OSError: If the source workbook cannot be read.

    Assumptions: workbook columns I:M hold the reported per-round coefficients;
    columns O:P hold the merged relative activity and its source SD. The source
    ``k`` is retained without conversion to kcat, Km or a calibrated kinetic unit.
    Raw time values are seconds; fluorescence is in instrument arbitrary units.
    Repeated source blocks are retained and audited, never silently deduplicated.
    No files are written. Figure 6 and the QF methods in the primary paper define
    the common Mca/Dpa chemistry and assay conditions.
    """
    path = Path(path)
    logger.info('Reading original PICS QF measurements from %s', path)
    rounds, summaries, traces = [], [], []
    with pd.ExcelFile(path) as workbook:
        for enzyme, code in MMP_TO_MEROPS.items():
            frame = pd.read_excel(workbook, sheet_name=enzyme, header=None)
            seen = {label: 0 for label in SUBSTRATE_LABELS}
            for row, value in frame.iloc[:, 8].items():
                if not isinstance(value, str) or value.strip() not in seen:
                    continue
                label = value.strip()
                seen[label] += 1
                t1, t2, coefficient, relative = map(float, frame.iloc[row, 9:13])
                if not np.isfinite([t1, t2, coefficient, relative]).all() or t1 >= t2:
                    raise ValueError(f'Invalid QF coefficient block: {enzyme}/I{row + 1}')
                rounds.append({
                    'enzyme': enzyme, 'merops_code': code, 'substrate': label,
                    'analysis_round': seen[label], 't1_s': t1, 't2_s': t2,
                    'source_k': coefficient, 'relative_activity_percent': relative,
                    'source_file': path.name, 'sheet': enzyme, 'source_k_cell': f'L{row + 1}',
                })
                if seen[label] == 1:
                    mean, sd = map(float, frame.iloc[row, 14:16])
                    if not np.isfinite([mean, sd]).all() or sd < 0:
                        raise ValueError(f'Invalid source summary: {enzyme}/O{row + 1}')
                    summaries.append({
                        'enzyme': enzyme, 'merops_code': code, 'substrate': label,
                        'relative_activity_percent': mean, 'source_sd_percent': sd,
                        'source_file': path.name, 'sheet': enzyme,
                        'source_mean_cell': f'O{row + 1}', 'source_sd_cell': f'P{row + 1}',
                    })
            if set(seen.values()) != {3}:
                raise ValueError(f'Expected three source analysis rounds per substrate: {enzyme}')

            # Read each source time row and its measured duplicate wells.
            time_rows = frame.index[frame.iloc[:, 3].eq('t')].tolist()
            if len(time_rows) != 3:
                raise ValueError(f'Expected three source time-series blocks: {enzyme}')
            for round_index, start in enumerate(time_rows, start=1):
                stop = time_rows[round_index] if round_index < len(time_rows) else len(frame)
                time = pd.to_numeric(frame.iloc[start, 4:], errors='coerce').to_numpy(dtype=float)  # (T,)
                current_substrate = None
                for row in range(start + 1, stop):
                    well = frame.iat[row, 1]
                    if not isinstance(well, str) or not re.fullmatch(r'[A-H]\d{2}', well):
                        continue
                    value = frame.iat[row, 0]
                    if isinstance(value, str):
                        current_substrate = next((s for s in SUBSTRATE_LABELS if s.replace('↓', '') == value.strip()), None)
                    if current_substrate is None:
                        raise ValueError(f'Unresolved QF well substrate: {enzyme}/{row + 1}')
                    intensity = pd.to_numeric(frame.iloc[row, 4:], errors='coerce').to_numpy(dtype=float)  # (T,)
                    if intensity.shape != time.shape or np.any(np.isfinite(intensity) & ~np.isfinite(time)):
                        raise ValueError(f'Intensity without a source time: {enzyme}/{row + 1}')
                    for column in np.flatnonzero(np.isfinite(time) & np.isfinite(intensity)):
                        traces.append({
                            'enzyme': enzyme, 'substrate': current_substrate, 'analysis_round': round_index,
                            'well': well, 'time_s': time[column], 'fluorescence_au': intensity[column],
                            'source_file': path.name, 'sheet': enzyme, 'excel_row': row + 1,
                            'excel_column': int(column + 5),
                        })
            logger.debug('Read QF coefficient and trace blocks for %s', enzyme)

    # Chemistry is transcribed from primary-paper Figure 6 and its adjacent text.
    substrate_rows = []
    for label in SUBSTRATE_LABELS:
        left, right = label.split('↓')
        tokens = (None, *left, right, 'Dpa', 'A', 'R')
        substrate_rows.append({
            'substrate': label, 'full_source_chemistry': f'Mca-{left}↓{right}-Dpa-AR-NH2',
            'source_motif': left + right, **dict(zip(POSITIONS, tokens)),
            'n_cap': 'Mca', 'c_cap': 'NH2', 'resolved_residues': 7,
            'publication_doi': PRIMARY_DOI, 'chemistry_locator': 'Figure 6A and accompanying text; printed pp. 47–49',
        })
    round_frame = pd.DataFrame(rounds)
    summary_frame = pd.DataFrame(summaries)
    trace_frame = pd.DataFrame(traces)
    audit = []
    for enzyme in MMP_TO_MEROPS:
        first = trace_frame.loc[trace_frame.enzyme.eq(enzyme) & trace_frame.analysis_round.eq(1)]
        third = trace_frame.loc[trace_frame.enzyme.eq(enzyme) & trace_frame.analysis_round.eq(3)]
        keys = ['substrate', 'well', 'time_s']
        a = first.set_index(keys).fluorescence_au.sort_index()
        b = third.set_index(keys).fluorescence_au.sort_index()
        audit.append({'enzyme': enzyme, 'rounds_1_and_3_identical': a.equals(b),
                      'source_analysis_rounds': 3, 'source_wells': trace_frame.loc[trace_frame.enzyme.eq(enzyme), ['analysis_round', 'well']].drop_duplicates().shape[0]})
    logger.info('Read %s source QF coefficient rows and %s reported summaries', len(round_frame), len(summary_frame))
    return {'rounds': round_frame, 'summary': summary_frame, 'traces': trace_frame,
            'substrates': pd.DataFrame(substrate_rows), 'round_audit': pd.DataFrame(audit)}


def qf_measurements_with_windows(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Associate reported QF coefficients and activities with exact source cut windows.

    :param data: Frames returned by :func:`read_qf_assay`.
    :type data: dict[str, pandas.DataFrame]
    :return: One row per reported coefficient or merged relative activity, with
        P4–P4′ tokens, chemistry, assay conditions and source cell locators.
    :rtype: pandas.DataFrame
    :raises ValueError: If a source label lacks exactly one chemistry definition.

    No activity threshold or coefficient fitting is applied. The unit of source
    ``k`` remains missing because its column does not specify a calibrated unit.
    P4 is absent, Dpa is retained at P2′, and terminal labels are kept separately.
    """
    chemistry = data['substrates']
    if chemistry.substrate.duplicated().any():
        raise ValueError('QF substrate chemistry definitions must be unique')
    coefficients = data['rounds'].rename(columns={'source_k': 'value', 'source_k_cell': 'source_value_cell'}).copy()
    coefficients['measurement_type'] = 'source_k'
    coefficients['unit'] = None
    coefficients['unit_status'] = 'not_reported'
    coefficients['value_sd'] = np.nan
    activities = data['summary'].rename(columns={
        'relative_activity_percent': 'value', 'source_mean_cell': 'source_value_cell',
        'source_sd_percent': 'value_sd',
    }).copy()
    activities['measurement_type'] = 'relative_activity'
    activities['unit'] = 'percent_of_PLG_reference'
    activities['unit_status'] = 'reported'
    combined = pd.concat([coefficients, activities], ignore_index=True)
    result = combined.merge(chemistry, on='substrate', how='left', validate='many_to_one')
    if result.full_source_chemistry.isna().any():
        raise ValueError('QF coefficient or activity lacks a source chemistry definition')
    result['source'] = 'PICS_QF'
    result['site_evidence'] = 'source_cut_annotation'
    result['cut_after_residue'] = 3
    result['ph'] = 7.5
    result['temperature_c'] = 37.0
    result['substrate_concentration_um'] = 1.0
    result['enzyme_concentration_min_nm'] = 1.0
    result['enzyme_concentration_max_nm'] = 10.0
    result['window'] = result[list(POSITIONS[:4])].apply(
        lambda row: ' '.join('-' if pd.isna(token) else token for token in row), axis=1
    ) + ' | ' + result[list(POSITIONS[4:])].agg(' '.join, axis=1)
    result['source_record_id'] = 'mmc8.xlsx/' + result.sheet + '/' + result.source_value_cell
    logger.info('Associated %s QF measurements with their source-defined windows', len(result))
    return result
