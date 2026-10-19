"""Small, source-specific corrections checked against primary publications."""
import math


def correct_peplife_context(record):
    """Correct a reviewed ELISA preparation/PK context mix-up; retain raw JSON."""
    expected = {'5066': 5040.0, '5067': 23760.0, '5068': 5040.0, '5069': 23760.0}
    identifier = record['source_record_id']
    if identifier not in expected or record['publication_id'] != '33057063':
        return record
    if record['relation'] != 'eq' or not math.isclose(record['half_life_seconds'], expected[identifier]):
        raise ValueError(f'Reviewed PEPlife2 context correction needs rechecking: {identifier}')
    if record['incubation_time_raw'] != 'Overnight at 4 °C':
        raise ValueError(f'PEPlife2 incubation context changed; recheck reviewed correction: {identifier}')
    record['reported_assay_mode'] = record['in_vivo_in_vitro']
    record['in_vivo_in_vitro'] = 'in vivo'
    record['temperature_c'] = None
    record['temperature_metadata_scope'] = 'detector_preparation_not_half_life'
    record['metadata_curation_note'] = ('Mode corrected to in vivo (mouse PK, Table 3). '
        '4°C overnight and 10 µg/mL describe ELISA plate coating, not degradation incubation or peptide exposure; temperature/concentration remain None.')
    record['curation_reference'] = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7566517/'
    return record
