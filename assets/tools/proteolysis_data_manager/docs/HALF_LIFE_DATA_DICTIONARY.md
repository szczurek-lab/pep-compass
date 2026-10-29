# Half-time two-table contract

`adapters/schema.py` defines the ordered fields. All text is UTF-8, TSV uses tab
separators and normal CSV quoting, and empty cells denote missing data. Readers
must preserve identifiers as strings and must not interpret a sequence such as
`NA` as a missing token. `datasets.read_half_time_datasets` enforces the contract.

## Peptide registry: sequences_half_time.tsv

| Field | Type | Meaning |
|---|---|---|
| sequence_id | string, primary key | SHA-256 of the normalized reported sequence and five chemistry fields |
| sequence | string | Whitespace/HTML-cleaned notation; case preserved for stereochemistry |
| chirality | nullable string | Reported stereochemistry; `mix` normalized to `mixed` |
| topology | nullable string | Reported linear/cyclic/branched description, including source detail |
| n_terminal_modification | nullable string | Reported N-terminal chemistry |
| c_terminal_modification | nullable string | Reported C-terminal chemistry |
| other_modifications | nullable string | Other reported chemical modifications |

Missing chemistry and explicitly unmodified (`free`) are different. Identity
merging requires equal descriptions, not sequence equality alone. Identical
incomplete descriptions can share a key; this is not proof that unknown
chemistry matches. No SMILES, descriptor or sequence-length columns are added.

## Source observations: half_time_experiments.tsv

| Fields | Type | Meaning |
|---|---|---|
| experiment_id | string, primary key | Source + author view + record ID + readout suffix where applicable |
| sequence_id | string, foreign key | Peptide registry reference |
| source, source_record_id | string | Database and original source record |
| publication_id, source_locator | nullable string | Publication identifier and original file/figure/record location |
| species, matrix, in_vivo_in_vitro | nullable string | Reported biological sample and experimental mode |
| matrix_fraction | nullable float | Fraction v/v, between 0 and 1 |
| temperature_c, ph | nullable float | Explicit assay conditions; pI is not assay pH |
| peptide_concentration_um | nullable float | Explicit molar concentration in µM; doses/mass concentrations are not guessed |
| assay | nullable string | Reported analytical method |
| measurement_kind | string | Half-life, remaining-fraction readout, or functional activity |
| relation | string | `eq`, `lt`, `le`, `gt`, `ge`, `approx`, `interval`, `unknown` |
| half_life_seconds | nullable float | Reported point; approximate points require `relation=approx` |
| lower_bound_seconds, upper_bound_seconds | nullable float | Reported censoring threshold or ordinary range, not CI |
| ci_lower_seconds, ci_upper_seconds, ci_level | nullable float | Explicit reported confidence limits and probability level |
| uncertainty_kind, uncertainty_value_seconds | nullable string / float | Reported SD/SE or untyped ± magnitude; no invented interpretation |
| replicate_count | nullable integer | Reported number of experiments/replicates, not inferred donor count |

Ordinary ranges, censored observations and CI do not generate a midpoint label.
Non-half-life readouts remain unlabelled in the numerical half-life fields; their
full original description is available in raw data by source record reference.
Repeated source representations are retained and are not independent replication.

## Reused source mapping and reviewed exceptions

- PEPlife2: both API partitions, original half-life/unit fields and recorded chemistry.
- PepMSND: published workbook labels and context; its SMILES/RDKit columns are not exported.
- PlifePred: author natural/mixed views; experimental context recovered only through the original ID + name + recorded chemistry rule. Author labels are retained.
- DRAMP: explicit condition-specific half-life clauses become separate observations. Mode recovery requires a consistent publication + chemistry + species/matrix match.
- Cavaco: 16 printed Figure 3a half-lives and 95% CI, checked against PDF digest and XML sequences; human serum, in vitro, 37 °C, 50% v/v, 500 µM, n=3.
- PEPlife2 5066–5069: the existing primary-publication correction retains mouse in-vivo PK context; ELISA coating temperature/concentration do not become assay exposure conditions.

Units, prose parsing, missingness and condition extraction reuse the existing
helpers. The migration snapshots and original notes remain in
`tmp/archive_old_analysis/half_time_two_table_migration` and the preceding archive.
