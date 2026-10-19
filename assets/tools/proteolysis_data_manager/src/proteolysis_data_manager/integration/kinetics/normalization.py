"""Build a complete BRENDA bundle; selection belongs to the protocol."""
from pathlib import Path

from proteolysis_data_manager.acquisition.kinetics.brenda import verify_snapshot
from proteolysis_data_manager.integration.kinetics.adapters import brenda
from proteolysis_data_manager.integration.kinetics.reader import KineticsDataset
from proteolysis_data_manager.integration.kinetics.schema import arrow_table
from proteolysis_data_manager.utils.io import sha256


def normalize_brenda(root):
    root = Path(root)
    receipt = verify_snapshot(root)
    rows, audit = brenda.load(root / receipt['raw_path'])
    dataset = KineticsDataset({name: arrow_table(name, records).to_pandas() for name, records in rows.items()}).validate()
    metadata = {
        'source': 'brenda', 'source_version': receipt['source_version'],
        'raw_sha256': receipt['sha256'], 'raw_path': receipt['raw_path'],
        'download_manifest_sha256': sha256(root / 'manifests/download_brenda.json'),
        'raw_kinetic_entries': audit['raw_entries'], 'parameter_counts': audit['parameter_counts'],
        'ec_blocks_with_measurements': audit['ec_blocks_with_measurements'],
        'experiment_policy': 'One source entry context; distinct entries are not linked by PMID/name/value coincidence.',
        'substrate_policy': 'Exact source label and explicit chemistry scoped to EC; not a resolved molecular identity.',
        'site_policy': 'Structured reported sites only; BRENDA comments/products are not searched or localized.',
        'unit_policy': {'turnover_number': 's^-1 unchanged', 'km_value': 'mM -> M: multiply by 0.001',
                        'kcat_km_value': '(mM*s)^-1 -> (M*s)^-1: multiply by 1000'},
        'unit_references': brenda.UNIT_REFERENCES,
        'implementation_sha256': {str(p.relative_to(Path(__file__).parent)): sha256(p)
                                  for p in sorted(Path(__file__).parent.rglob('*.py'))},
    }
    manifest = dataset.write(root / 'normalized', root / 'manifests/normalization_brenda.json', metadata=metadata)
    notes = [
        '# BRENDA normalization decisions', '',
        f"Raw kinetic entries: {audit['raw_entries']}; retained measurements: {len(dataset['kinetic_measurements'])}.", '',
        '- Every EC 3.4 kinetic entry survives; no organism, sequence, site or MEROPS-overlap gate.',
        '- Experiment IDs identify source contexts, not independently established experiments or replicates.',
        '- Enzyme accessions/organisms come from the linked source protein block, never an EC-to-MEROPS panel.',
        '- Multiple protein links stay unresolved; they do not multiply measurements.',
        '- Exact substrate labels are grouped within an EC and explicit chemistry; they are not resolved sequences.',
        '- Chemistry is retained only when explicitly reported; no enrichment from half-life/cut-window datasets.',
        '- Conditions come from an unambiguous scalar in that entry comment. Optimum fields are not assay conditions.',
        '- Matrix, sample species and in-vitro mode remain null until the experimental setup is documented.',
        '- Source comment and original value are available at each JSON locator in the archived raw file.',
        '- No BRENDA free-text site localization is attempted. Empty site tables mean no structured annotations extracted.',
        '- Missing SD/CI/replicate counts remain null; no estimator or range midpoint is used.',
        '- kcat: s^-1; Km: mM multiplied by 0.001; kcat/Km: inverse mM/s multiplied by 1000.',
        '- BRENDA datafields has an inconsistent kcat/Km heading (mM/s); the search form uses inverse mM and time.',
        '  Unit evidence: ' + brenda.UNIT_REFERENCES['kcat_km'], '',
        '## Entries requiring attention', '',
    ]
    for note in audit['notes']:
        notes.append(f"- `{note['locator']}` (`{note['measurement_id']}`): " + '; '.join(note['reasons']))
    (root / 'normalized/remarks_brenda.md').write_text('\n'.join(notes) + '\n')
    return dataset, manifest
