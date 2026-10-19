"""Concise per-source normalization decisions, regenerated beside the Parquet."""
from pathlib import Path
import pandas as pd
from proteolysis_data_manager.integration.half_life.schema import plain


def _cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def _table(columns, rows):
    if not rows:
        return 'None.\n'
    return '| ' + ' | '.join(columns) + ' |\n|' + '|'.join(['---'] * len(columns)) + '|\n' + ''.join(
        '| ' + ' | '.join(_cell(v) for v in row) + ' |\n' for row in rows)


def write_remarks(inventory, audit, root):
    """Record exceptions, retained bounds and field-level enrichment sources."""
    folder = Path(root) / 'normalized'; folder.mkdir(parents=True, exist_ok=True)
    indexed = audit.set_index('observation_id')
    for source, data in inventory.groupby('source', sort=True):
        notes, enrichment, conflicts = [], [], []
        for record in data.to_dict('records'):
            identifier = record['observation_id']
            a = indexed.loc[identifier]
            label = plain(record['duration_class'])
            point = plain(record['half_life_seconds'])
            relation = record['relation']
            raw = record.get('measurement_clause_raw') or record['half_life_raw']
            if record['unit_raw']:
                raw += ' ' + record['unit_raw']
            publication = record.get('publication_id')
            ref = f'[PMID {publication}](https://pubmed.ncbi.nlm.nih.gov/{publication}/)' if publication else 'Raw source only'
            if record['duration_class_status'] == 'project_heuristic_reviewed':
                decision = 'medium; numeric=None. Project heuristic, not an author class. Paper confirms >1 h in vivo; >6 h remains possible. Excluded from in-vitro protocols.'
                ref = '[Zhang 2014, abstract/highlights](https://doi.org/10.1016/j.biocel.2014.07.004)'
            elif relation != 'eq':
                if label:
                    decision = f'{label}; numeric=None. Reported bound/range lies entirely within this project class.'
                elif relation in {'lt', 'le', 'gt', 'ge', 'interval'}:
                    decision = f'class=None; numeric=None. Compatible classes: {record["duration_class_candidates"]}; no unsupported choice or midpoint.'
                elif relation == 'approx':
                    decision = 'class=None; numeric=None. Approximation has no documented tolerance interval.'
                elif record['measurement_kind'] != 'half_life':
                    decision = 'class=None; numeric=None. Activity/fraction at a time is not a measured half-life.'
                else:
                    decision = 'class=None; numeric=None. Parser unresolved; source review required.'
            elif pd.notna(record['reported_error_seconds']):
                decision = f'Point={point:g} s; {label}. ± error={record["reported_error_seconds"]:g} s; type unspecified, SD/CI remain None.'
            else:
                decision = None
            if decision:
                notes.append([identifier, raw, decision, ref])
            if isinstance(record.get('metadata_curation_note'), str):
                notes.append([identifier, 'Raw mode=' + str(record.get('reported_assay_mode')) + '; ' + str(record.get('incubation_time_raw')),
                              record['metadata_curation_note'], '[Reviewed primary paper: Table 3; In vivo PK assays](' + record['curation_reference'] + ')'])
            if record.get('linked_label_status') in {'precision_or_censoring_changed', 'value_conflict', 'requires_review'}:
                conflicts.append([identifier, record['linked_label_status'], 'Historical author label retained; not replaced by current PEPlife2 label.', ref])
            for field in a['enriched_fields']:
                enrichment.append([identifier, field, ', '.join(a['field_sources'][field])])
            if record.get('metadata_origin') == 'peplife2_id_and_chemistry_match':
                enrichment.append([identifier, 'species/matrix/mode; experimental context',
                                   'peplife2:all:' + str(record['linked_peplife2_id'])])
            elif record.get('metadata_origin') == 'peplife2_publication_and_chemistry_match':
                enrichment.append([identifier, 'in_vivo_in_vitro (consistent linked context)',
                                   'PEPlife2 candidate IDs: ' + str(record.get('candidate_peplife2_ids', ''))])
            if a['conflicting_fields']:
                conflicts.append([identifier, ', '.join(a['conflicting_fields']), 'Known values disagree: separate normalized observations retained.', ref])
            if record.get('ci_origin', 'not_reported') != 'not_reported' and point is None:
                notes.append([identifier, raw, 'Representative numeric value = midpoint of explicitly documented CI; not a measured mean.', ref])
        content = f'# {source} — normalization remarks\n\n'
        content += 'Project classes: short <30 min; medium 30 min–6 h inclusive; long >6 h. Missing CI/SD stay None. No synthetic uncertainty.\n\n'
        content += 'IDs use `source:view:record_id[:measurement_n]`; raw files remain in `../raw/' + source + '/`. Unlisted exact points require only unit conversion.\n\n'
        content += 'PMID links are source-provided references; specifically reviewed primary-paper decisions are marked explicitly.\n\n'
        if source == 'plifepred':
            content += 'Context enrichment: original PEPlife ID + peptide name + full recorded chemistry must match. Historical labels remain unchanged.\n\n'
        if source == 'dramp':
            content += 'Multiple explicit half-life/matrix clauses become separate observations. Mode recovery requires a unique consistent PEPlife publication + chemistry + species/matrix match.\n\n'
        if source == 'cavaco':
            content += '16 printed Figure 3a validation points; reported 95% CI, n=3. Verified PDF hash and Table 1 sequences. 1:1 serum mixing → fraction=0.5; final peptide concentration=500 µM. See checked-in Cavaco transcription provenance.\n\n'
        content += '## Label decisions\n\n' + _table(['Record', 'Source wording', 'Decision / reason', 'Reference'], notes)
        content += '\n## Shared-record field enrichment\n\n' + _table(['Receiving record', 'Field', 'Donor record(s)'], enrichment)
        content += '\n## Conflicts retained separately\n\n' + _table(['Record', 'Conflict', 'Decision', 'Reference'], conflicts)
        (folder / f'remarks_{source}.md').write_text(content)
