"""Reviewed printed Cavaco validation points and documented CI."""
import json
import hashlib
import pandas as pd
from proteolysis_data_manager.integration.half_life.adapters.common import row

def load(raw_root, config_dir):
    rows = []
    # Printed numerical labels (not digitized curve estimates); source and
    # transcription details are checked in with the acquisition configuration.
    curated = pd.read_csv(config_dir / 'cavaco_validation.tsv', sep='\t', keep_default_na=False)
    provenance = json.loads((config_dir / 'cavaco_validation.provenance.json').read_text())
    pdf_digest = hashlib.sha256((raw_root / 'cavaco' / 'article.pdf').read_bytes()).hexdigest()
    if pdf_digest != provenance['article_pdf_sha256']:
        raise ValueError('Cavaco PDF differs from the reviewed numerical transcription; review the curated table first.')
    import xml.etree.ElementTree as ET
    article = ET.parse(raw_root / 'cavaco' / 'article.xml').getroot()
    expected_sequences = {}
    for table in article.findall('.//table-wrap'):
        if table.findtext('label') == 'Table 1':
            for tr in table.findall('.//tbody/tr'):
                cells = [''.join(cell.itertext()).strip() for cell in tr]
                expected_sequences[int(cells[0])] = cells[1]
    if dict(zip(curated.peptide_id, curated.sequence)) != expected_sequences:
        raise ValueError('Cavaco transcription does not match the 16 sequences in Table 1.')
    if not ((curated.ci_lower_minutes > 0) & (curated.ci_lower_minutes <= curated.half_life_minutes) & (curated.half_life_minutes <= curated.ci_upper_minutes)).all():
        raise ValueError('Invalid confidence interval in Cavaco transcription.')
    for raw in curated.to_dict('records'):
        r = row('cavaco', 'experimental_validation_16', raw['peptide_id'], raw, raw['sequence'], raw['half_life_minutes'], 'minutes',
                species='human', matrix='serum', in_vivo_in_vitro='in vitro', topology='linear', chirality='l',
                n_terminal_modification='free', c_terminal_modification='amidation', chemical_modifications='free',
                publication_id='33641212', assay='RP-HPLC; LC-MS', temperature_c=37.0, matrix_fraction=0.5,
                peptide_concentration_raw='0.5 mM after 1:1 mixing', n_replicates=3,
                ci_lower_seconds=float(raw['ci_lower_minutes'])*60, ci_upper_seconds=float(raw['ci_upper_minutes'])*60,
                source_filename='article.pdf', source_locator='Table 1; Figure 3a (printed labels); Methods',
                metadata_origin='reviewed_transcription_of_printed_values',
                half_life_mean_seconds=float(raw['half_life_minutes'])*60,
                half_life_estimate_type='reported_mean_of_fitted_half_lives',
                ci_confidence_level=.95, ci_origin='reported_figure_3a', uncertainty_kind='reported_ci')
        rows.append(r)
    return rows
