"""DRAMP prose clauses, split into condition-specific observations."""
import re
from proteolysis_data_manager.integration.half_life.adapters.common import row, text, medium, chemistry_field, excel_rows, NUMBER, UNIT

def dramp_rows(raw):
    """Split only explicit half-life clauses with their own matrix description.

    Original source text remains attached. Two timepoints from a degradation
    curve are never split into independent half-life observations.
    """
    original = text(raw['half_life'])
    parts = re.split(r'[,;]\s*(?=(?:[<>]=?\s*)?\d)', original)
    explicit = re.search(r'half[ -]?li(?:fe|ves|ve)|t1/2', original, re.I)
    split = bool(explicit and len(parts) > 1 and all(re.search(r'\b(?:serum|plasma)\b', part, re.I) and
                 re.search(NUMBER+r'(?:\s*±\s*'+NUMBER+r')?\s*'+UNIT+r'\b', part, re.I) for part in parts))
    clauses = parts if split else [original]
    result = []
    fallback_species, fallback_matrix = medium(raw['Type'])
    for i, clause in enumerate(clauses, 1):
        input_text = clause if re.search(r'half[ -]?li(?:fe|ves|ve)', clause, re.I) else ('Half-life is '+clause if split else clause)
        species, matrix = medium(clause)
        if species == 'unknown': species = fallback_species
        if matrix == 'other': matrix = fallback_matrix
        pct = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:aqueous\s+)?(?:human\s+|mouse\s+|rat\s+)?(?:serum|plasma)', clause, re.I)
        fraction = float(pct[1])/100 if pct and 0 < float(pct[1]) <= 100 else (1.0 if re.search(r'\bfull\s+(?:human\s+|mouse\s+|rat\s+)?(?:serum|plasma)', clause, re.I) else None)
        r = row('dramp', 'stability', raw['DRAMP_ID'], raw, raw['Sequence'], input_text, prose=True,
                species=species, matrix=matrix, test_sample_raw=text(raw['Type']), name=text(raw.get('Name')),
                publication_id=text(raw.get('Pubmed_ID')), assay=text(raw.get('Assay')),
                chirality=text(raw.get('Stereochemistry')).lower() or 'unknown',
                topology=text(raw.get('Linear/Cyclic/Branched')).lower() or 'unknown',
                n_terminal_modification=chemistry_field(raw.get('N-terminal_Modification')),
                c_terminal_modification=chemistry_field(raw.get('C-terminal_Modification')),
                chemical_modifications=chemistry_field(raw.get('Other_Modifications')), source_filename='stability_amps.xlsx',
                matrix_fraction=fraction, measurement_index=i, measurement_clause_raw=clause)
        r['half_life_raw'] = original
        if split: r['observation_id'] += f':measurement_{i}'
        result.append(r)
    return result


def load(raw_root):
    return [record for raw in excel_rows(raw_root / 'dramp' / 'stability_amps.xlsx', {'DRAMP_ID', 'Sequence', 'half_life', 'Type'}) for record in dramp_rows(raw)]
