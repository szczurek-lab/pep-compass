"""Published PepMSND workbook, independently normalized."""
from proteolysis_data_manager.integration.half_life.adapters.common import row, text, medium, chemistry_field, excel_rows

def load(raw_root):
    rows = []
    for raw in excel_rows(raw_root / 'pepmsnd' / 'dataset.xlsx', {'ID', 'Sequence', 'T1/2(hour)', 'Test Sample', 'In-Vivo/Vitro'}):
        species, matrix = medium(raw['Test Sample'])
        rows.append(row('pepmsnd', 'published', raw['ID'], raw, raw['Sequence'], raw['T1/2(hour)'], 'hours',
                        species=species, matrix=matrix, test_sample_raw=text(raw['Test Sample']), name=text(raw.get('Name')),
                        in_vivo_in_vitro=text(raw['In-Vivo/Vitro']).lower(), smiles=text(raw.get('SMILES')),
                        topology=text(raw.get('Cyclic/Linear')).lower() or 'unknown',
                        n_terminal_modification=chemistry_field(raw.get('N-ter MOD')), c_terminal_modification=chemistry_field(raw.get('C-ter MOD')),
                        chemical_modifications=chemistry_field(raw.get('CHEMICAL MODIFICATION')), source_filename='dataset.xlsx'))
    return rows
