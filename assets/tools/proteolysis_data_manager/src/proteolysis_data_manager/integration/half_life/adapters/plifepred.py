"""PlifePred author subsets with explicit PEPlife identity links."""
import math
from proteolysis_data_manager.integration.half_life.adapters.common import row, text, chemistry_field, chemical_signature, excel_rows

def load(raw_root, peplife_by_id):
    rows = []
    for view, column in [('natural', 'hl in seconds'), ('mixed', 'Half-life in seconds')]:
        for raw in excel_rows(raw_root / 'plifepred' / f'{view}.xlsx', {'id.pdb', 'seq', column}):
            identifier = str(raw['id.pdb']).removesuffix('.pdb')
            r = row('plifepred', view, identifier, raw, raw['seq'], raw[column], 'seconds', name=text(raw.get('name')),
                    chirality=text(raw.get('chiral')).lower() or 'unknown', topology=text(raw.get('lin_cyc')).lower() or 'unknown',
                    n_terminal_modification=chemistry_field(raw.get('nter')), c_terminal_modification=chemistry_field(raw.get('cter')),
                    chemical_modifications=chemistry_field(raw.get('chem_mod')), source_filename=f'{view}.xlsx')
            candidate = peplife_by_id.get(identifier)
            # Recover matrix/publication only through the author ID plus full
            # recorded chemistry and peptide name. Label agreement is reported
            # separately; historical benchmark labels are never overwritten.
            if candidate and chemical_signature(r) == chemical_signature(candidate) and r['name'] == candidate['name']:
                for field in ('species', 'matrix', 'in_vivo_in_vitro', 'publication_id', 'assay', 'test_sample_raw', 'peptide_concentration_raw', 'incubation_time_raw'):
                    r[field] = candidate[field]
                r['metadata_origin'] = 'peplife2_id_and_chemistry_match'
                r['linked_peplife2_id'] = identifier
                if candidate['relation'] != r['relation']:
                    r['linked_label_status'] = 'precision_or_censoring_changed'
                elif r['relation'] == 'eq':
                    r['linked_label_status'] = 'same_label' if math.isclose(r['half_life_seconds'], candidate['half_life_seconds'], rel_tol=1e-6) else 'value_conflict'
                else:
                    r['linked_label_status'] = 'requires_review'
            rows.append(r)
    return rows
