"""Extract qPISA measurements with source provenance and masked MEROPS windows."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from proteolysis_data_manager.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)
POSITIONS = ('P4', 'P3', 'P2', 'P1', 'P1p', 'P2p', 'P3p', 'P4p')
DOI = '10.1038/s44320-024-00071-4'
EV1_FILE = '44320_2024_71_MOESM2_ESM.xlsx'
EV4_FILE = '44320_2024_71_MOESM5_ESM.xlsx'
EV6_FILE = '44320_2024_71_MOESM7_ESM.xlsx'


def _attach_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach P4–P4′ tokens from resolved parent sequences, cutting after residue two.

    :param frame: Records with ``substrate_sequence``; missing parents stay missing.
    :return: New frame with positional columns, display label and resolved count.
    """
    result = frame.copy()
    sequence = result['substrate_sequence'].astype('string')
    resolved = sequence.notna()
    for position in POSITIONS[:2]:
        result[position] = None
    for offset, position in enumerate(POSITIONS[2:]):
        result[position] = sequence.str[offset].where(resolved, None).astype(object)
    result['has_window'] = resolved
    result['resolved_positions'] = np.where(resolved, 6, 0)
    result['window'] = ('--' + sequence.str[:2] + '|' + sequence.str[2:6]).where(resolved)
    result['cut_after_residue'] = pd.array(np.where(resolved, 2, np.nan), dtype='Int64')
    return result


def read_endpoint_measurements(directory: str | Path) -> pd.DataFrame:
    """Read all EV1/EV4 endpoint measurements, preserving positive and negative changes.

    :param directory: Original ``raw/dpp4_qpisa`` directory.
    :type directory: str | pathlib.Path
    :return: One record per original peptide, enzyme and experiment; input and
        product records retain their respective measurement types and provenance.
    :rtype: pandas.DataFrame
    :raises ValueError: If source headers, peptide syntax or numeric levels differ
        from the supported source layout.
    :raises OSError: If an original workbook is unavailable.

    Substrate depletion is control minus enzyme in normalized log2 units.
    Product enrichment uses the opposite sign. No threshold or rate fitting is
    applied. Product parents are assigned only when exactly one deposited input
    sequence equals two residues followed by the product. This is a sequence
    association, not an experimentally supplied parent–product pairing.
    Original files are never changed. Conditions are from the article methods.
    """
    directory = Path(directory)
    outputs = []
    sources = (
        ('EV1', EV1_FILE, 'Peptide_Levels_dpf-3_dpp4', 'Peptide squence',
         (('DPF-3', 'DPF-3', 'No enzyme control (for comparing with DPF-3 digested samples)'),
          ('DPP4', 'DPP4', 'No enzyme control (for comparing with DPP4 digested samples)'))),
        ('EV4', EV4_FILE, 'Peptide_Levels_dpf-3', 'Peptide sequence',
         (('DPF-3', 'DPF-3', 'No enzyme control'),)),
    )
    for experiment, filename, sheet, sequence_column, assays in sources:
        logger.info('Reading qPISA %s endpoint measurements', experiment)
        source = pd.read_excel(directory / filename, sheet_name=sheet)
        required = {sequence_column, 'Status', *(column for _, a, b in assays for column in (a, b))}
        if set(source.columns) != required or source[sequence_column].duplicated().any():
            raise ValueError(f'Unexpected qPISA headers or duplicate peptide sequences: {filename}')
        sequence = source[sequence_column]
        if not sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{6,}', na=False).all():
            raise ValueError(f'Unsupported qPISA sequence spelling: {filename}')
        if not source['Status'].isin(['Tryptic+', 'Non-Tryptic+']).all():
            raise ValueError(f'Unknown qPISA peptide status: {filename}')

        # Recover only unique deposited substrate candidates for product sequences.
        is_input = source['Status'].eq('Tryptic+')
        parents: dict[str, list[str]] = {}
        for peptide in sequence[is_input]:
            parents.setdefault(peptide[2:], []).append(peptide)
        candidate_count = sequence.map(lambda peptide: len(parents.get(peptide, ())))
        parent_sequence = sequence.where(is_input, sequence.map(
            lambda peptide: parents[peptide][0] if len(parents.get(peptide, ())) == 1 else None))
        evidence = pd.Series('unresolved_parent', index=source.index)
        evidence.loc[~is_input & candidate_count.gt(1)] = 'ambiguous_parent'
        evidence.loc[~is_input & candidate_count.eq(1)] = 'unique_sequence_match_in_deposit'
        evidence.loc[is_input] = 'enzyme_geometry'
        for enzyme, enzyme_column, control_column in assays:
            enzyme_level = pd.to_numeric(source[enzyme_column], errors='raise').astype(float)
            control_level = pd.to_numeric(source[control_column], errors='raise').astype(float)
            if not np.isfinite(enzyme_level).all() or not np.isfinite(control_level).all():
                raise ValueError(f'Non-finite qPISA endpoint levels: {filename}/{enzyme}')
            treated_to_control = np.exp2(enzyme_level - control_level)  # (N_peptides,)
            frame = pd.DataFrame({
                'source': 'qPISA', 'experiment': experiment, 'enzyme': enzyme,
                'merops_code': 'S09.003' if enzyme == 'DPP4' else None,
                'measured_sequence': sequence, 'status': source['Status'],
                'substrate_sequence': parent_sequence, 'site_evidence': evidence,
                'candidate_parent_count': candidate_count.where(~is_input, 1),
                'measurement_type': np.where(is_input, 'substrate_log2_depletion', 'product_log2_enrichment'),
                'value': np.where(is_input, control_level - enzyme_level, enzyme_level - control_level),
                'unit': 'log2_fold_change', 'enzyme_log2_level': enzyme_level,
                'control_log2_level': control_level, 'treated_to_control_signal_ratio': treated_to_control,
                'relative_signal_loss': (1 - treated_to_control).where(is_input),
                'duration_s': 14400, 'ph': 7.4, 'temperature_c': 21.0,
                'enzyme_mass_ug': 0.5 if enzyme == 'DPP4' else 5.0,
                'publication_doi': DOI, 'source_file': filename, 'source_sheet': sheet,
                'source_excel_row': source.index + 2,
                'enzyme_source_column': enzyme_column, 'control_source_column': control_column,
            })
            frame['source_record_id'] = (experiment + '/' + enzyme + '/' + frame['source_excel_row'].astype(str))
            outputs.append(_attach_windows(frame))
    result = pd.concat(outputs, ignore_index=True)
    logger.info('Extracted %s qPISA endpoint records; %s with windows', len(result), int(result.has_window.sum()))
    return result


def read_timecourse_measurements(directory: str | Path) -> dict[str, pd.DataFrame]:
    """Read EV6 source pairs and control-matched changes at the same time and replicate.

    :param directory: Original ``raw/dpp4_qpisa`` directory.
    :type directory: str | pathlib.Path
    :return: ``pairs`` with both measured source intensities and ``changes`` with
        substrate depletion and product enrichment relative to matched controls.
    :rtype: dict[str, pandas.DataFrame]
    :raises ValueError: If source pair labels, conditions, intensities or control
        matches are missing, ambiguous or inconsistent.
    :raises OSError: If the original workbook cannot be read.

    Substrate/product columns are paired by their original column positions;
    identical product sequences from different designed substrates stay separate.
    Controls are matched by assay pair, time and replicate. No DPP4 zero-minute
    observation is invented. Ratios are signal ratios, not calibrated amounts.
    """
    directory = Path(directory)
    sheet = 'Product peptide intensities'
    logger.info('Reading qPISA EV6 substrate/product time courses')
    source = pd.read_excel(directory / EV6_FILE, sheet_name=sheet, header=None)
    if source.shape[1] != 9:
        raise ValueError('Expected four source substrate/product column pairs in EV6')
    records = []
    for column in range(1, 9, 2):
        substrate = re.fullmatch(r'\s*([A-Z]+) \(substrate\)', str(source.iat[1, column]))
        product = re.fullmatch(r'\s*([A-Z]+) \(product\)', str(source.iat[1, column + 1]))
        if not substrate or not product or substrate[1][2:] != product[1]:
            raise ValueError(f'Unsupported EV6 source pairing at column {column + 1}')
        for row in range(2, len(source)):
            condition = re.fullmatch(r'(Buffer only|hDPP4)\. Replicate ([A-Z])\. (\d+) min', str(source.iat[row, 0]).strip())
            if condition is None:
                raise ValueError(f'Unsupported EV6 condition at row {row + 1}')
            substrate_signal, product_signal = map(float, source.iloc[row, column:column + 2])
            if not np.isfinite([substrate_signal, product_signal]).all() or min(substrate_signal, product_signal) <= 0:
                raise ValueError(f'Non-positive or non-finite EV6 signal at row {row + 1}')
            records.append({
                'source': 'qPISA', 'experiment': 'EV6', 'enzyme': 'DPP4', 'merops_code': 'S09.003',
                'assay_pair': column // 2 + 1, 'substrate_sequence': substrate[1], 'product_sequence': product[1],
                'condition': condition[1], 'replicate': condition[2], 'time_s': int(condition[3]) * 60,
                'substrate_signal_au': substrate_signal, 'product_signal_au': product_signal,
                'site_evidence': 'source_substrate_product_pair', 'temperature_c': 21.0,
                'publication_doi': DOI, 'source_file': EV6_FILE, 'source_sheet': sheet,
                'source_excel_row': row + 1, 'substrate_excel_column': column + 1,
                'product_excel_column': column + 2,
            })
    pairs = _attach_windows(pd.DataFrame(records))
    keys = ['assay_pair', 'time_s', 'replicate']
    if pairs.duplicated(['condition', *keys]).any():
        raise ValueError('Duplicate EV6 source conditions')
    control = pairs.loc[pairs.condition.eq('Buffer only'), keys + ['substrate_signal_au', 'product_signal_au', 'source_excel_row']]
    control = control.rename(columns={'substrate_signal_au': 'control_substrate_signal_au',
                                      'product_signal_au': 'control_product_signal_au',
                                      'source_excel_row': 'control_source_excel_row'})
    changes = pairs.loc[pairs.condition.eq('hDPP4')].merge(control, on=keys, how='left', validate='one_to_one')
    if changes.control_source_excel_row.isna().any():
        raise ValueError('EV6 enzyme observation lacks an exact time/replicate-matched control')
    changes['substrate_signal_ratio'] = changes.substrate_signal_au / changes.control_substrate_signal_au
    changes['substrate_log2_depletion'] = -np.log2(changes.substrate_signal_ratio)
    changes['relative_substrate_signal_loss'] = 1 - changes.substrate_signal_ratio
    changes['product_signal_ratio'] = changes.product_signal_au / changes.control_product_signal_au
    changes['product_log2_enrichment'] = np.log2(changes.product_signal_ratio)
    logger.info('Extracted %s EV6 source pairs and %s matched changes', len(pairs), len(changes))
    return {'pairs': pairs, 'changes': changes}
