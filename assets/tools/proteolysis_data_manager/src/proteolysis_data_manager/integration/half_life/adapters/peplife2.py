"""PEPlife2 JSON partitions."""
import json
from proteolysis_data_manager.integration.half_life.adapters.common import row, text, medium, chemistry_field
from proteolysis_data_manager.integration.half_life.adapters.curation import correct_peplife_context

def peplife_row(raw) -> dict:
    species, matrix = medium(raw.get('test_sample'))
    return correct_peplife_context(row('peplife2', 'all', raw['id'], raw, raw['seq'], raw['half_life'], raw['units_half'],
               species=species, matrix=matrix, name=text(raw.get('name')),
               in_vivo_in_vitro=text(raw.get('vivo_vitro')).lower() or 'unknown',
               publication_id=text(raw.get('pmid')), assay=text(raw.get('assay')),
               chirality=text(raw.get('chiral')).lower() or 'unknown', topology=text(raw.get('lin_cyc')).lower() or 'unknown',
               n_terminal_modification=chemistry_field(raw.get('nter')), c_terminal_modification=chemistry_field(raw.get('cter')),
               chemical_modifications=chemistry_field(raw.get('chem_mod')), peptide_concentration_raw=text(raw.get('conc')),
               test_sample_raw=text(raw.get('test_sample')), incubation_time_raw=text(raw.get('incubation_time'))))


def load(raw_root):
    rows = []
    for partition in ('natural', 'modified'):
        for raw in json.loads((raw_root / 'peplife2' / f'{partition}.json').read_text())['data']:
            record = peplife_row(raw)
            record['source_filename'] = f'{partition}.json'
            rows.append(record)
    return rows
