# Unified proteolysis dataset, schema 1.0

## Primary common dataset

The primary public dataset is `reader.to_frame()`: **named numeric feature columns**
on source-scoped assay/subject/window/time/replicate rows. The normalized tables
are its provenance backend. `quantity/value` views remain available for auditing;
they are not the final feature table. No source measurements are matched merely
because their peptide windows or condition descriptions coincide.

```python
from proteolysis_data_manager.integration.merge import ProteolysisDataset
reader = ProteolysisDataset.read("data/proteolysis/merged_datasets/proteolysis.sqlite")
merged = reader.to_frame()
# All three constants and the linked cut window are on the same eligible row.
kinetics = merged.loc[merged.kcat_km.notna(), [
    "source_db", "enzyme_name", "substrate_name", "window",
    "kcat", "km", "kcat_km", "kcat_km_basis",
]]
probabilities = reader.position_probabilities(frame=merged)
```

The public numerical column/unit registry is `contract/features.py:FEATURE_UNITS`.
All common feature columns have stable names; absent measurements remain null.
Each original quantity also has its measurement ID, source record, unit, relation,
bounds and derivation columns. A source parameter associated with several candidate
windows may appear on several rows with the **same measurement ID**. Those rows are
not independent measurements. Repeated parameters within one source group are kept
unpaired, with `parameter_pairing="multiple_values_unpaired"`; no averaging or
arbitrary matching is performed.

| Common column | Transformation / basis |
| --- | --- |
| `kcat`, `km`, `kcat_km` | Source-normalized SI parameters. Missing efficiency is derived as kcat/km only from a unique same-assay/subject pair, km > 0; reported efficiencies remain unchanged |
| `half_life` | Source half-life in seconds, retaining source limits separately |
| `remaining_fraction` | Reported fraction or 2^(-d), where d = log2(control/treated) |
| `substrate_remaining_percent` | 100 × remaining_fraction |
| `substrate_depletion_percent` | 100 × (1 − remaining_fraction); negative net changes are retained |
| `product_signal_increase_percent` | 100 × (2^e − 1), where e = log2(treated/control) |
| `apparent_rate_s_inv` | ln(2) × d/time for endpoint loss, or ln(2)/half_life; exponential-loss assumption recorded in `apparent_rate_basis` |
| `relative_activity_percent` | QF source normalization 100 × k/k_PLG within enzyme and source round; source merged mean/SD stay separate |
| `fluorescence_rate_au_s` | OLS slope of duplicate-well mean fluorescence over the source-selected t1..t2 interval; fit R² and endpoints retained |
| `p_P4` … `p_P4p` | Common empirical amino-acid occurrence probabilities at observed positions |
| `position_probability_sum` | Sum of available empirical positional probabilities, accompanied by coverage and profile source |
| `merops_probability_sum` | Separately retained original smoothed MEROPS reference score |

Control-relative signal loss, PLG-relative activity and absolute molecular conversion
have different reference bases. Those bases remain named in the table. A fluorescence
slope is not renamed kcat; no unreported fully-cleaved fluorescence calibration is
assumed. The percentage and apparent-rate transformations preserve all original
source values and do not impose a depletion threshold.

### Common positional distributions

`position_probabilities()` returns one row per source/profile/position, with 20
amino-acid columns and `observed_weight`. For MEROPS and PICS:

`p(a,j) = count(a,j) / sum_a count(a,j)`.

For qPISA EV1/EV4 parent-substrate windows, weights are
`max(0, 1 − 2^(-d))`; weighted counts are normalized in the same way. This uses
positive continuous depletion as candidate-cut evidence under the approximately
mixed-input assumption. Product changes and repeated EV6 time points do not add
occurrence counts. Detection and input-library biases are not corrected.

PICS enzymes and trypsin/GluC libraries remain separate. qPISA experiments/enzymes
remain separate, including DPF-3 without a MEROPS identifier. MEROPS populations
remain separately indexed. No smoothing or background correction is applied to
these empirical distributions. Zero positional depth remains null.

`position_probability_sum` uses a source's own PICS/qPISA distribution when
available, otherwise an explicitly matching MEROPS reference. Each row retains
`probability_source_db`, `probability_source_dataset`, weighting and position
coverage. These are descriptive marginal frequencies, including the query rows;
they are not independent validation scores or calibrated P(cleavage | sequence).
Unsupported residues retain null positional probabilities; the sum of available
terms is explicitly accompanied by observed and scored position counts.

## Module structure

```text
merge/
  contract/                   # backend schema, builder, dataset API, feature units
  parsers/
    half_life/                # one module per half-life dataset
    kinetics/                 # BRENDA, SABIO and shared cut recovery
    window_neighbourhood/     # MEROPS, PICS, QF and qPISA
  features/                   # activity transforms, frequencies, MEROPS references
  views/                      # wide table and overall statistics
  storage/                    # SQLite writer and reader
  build.py                    # source composition and CLI
  registry.py                 # parser registration
```

The old root `model`, `schema`, `reader` and `scoring` modules contain compatibility
imports only. Scientific implementation lives in the thematic packages. The SQLite
backend schema stays at 1.0 and existing acquired and merged files remain readable;
new common feature columns are computed by the reader without modifying raw files
or rewriting the existing database.

Two short notebooks present the actual outputs:

- `datasets_analysis/merge/0_01_common_parameters.ipynb`: common numeric values and
  overall source/feature availability.
- `datasets_analysis/merge/0_02_position_probabilities.ipynb`: common source
  distributions and a selectable 8 × 20 profile.


## Public API

```python
from proteolysis_data_manager.integration.merge import (
    ProteolysisDataset, build_dataset, parse_dataset, add_merops_scores, write_dataset,
)

# Source selection is explicit; omitted parsers contribute no empirical records.
dataset = build_dataset("data/proteolysis", sources=["brenda", "sabio"])
print(dataset.summary())

# A single parser accepts its acquired series root via the registry.
pics = parse_dataset("mmp_pics", "data/proteolysis")
pics = add_merops_scores(
    pics, "data/proteolysis/window_neighbourhood_datasets/raw/merops",
    dataset_index=34,
)

# Only this explicit call writes the common dataset; notebook conversion does not.
write_dataset(dataset, "data/proteolysis/merged_datasets/proteolysis.sqlite")
reader = ProteolysisDataset.read("data/proteolysis/merged_datasets/proteolysis.sqlite")
kinetics = reader.measurements(quantity="kcat")
windows = reader.contexts()
scores = reader.features()
values = reader.values()  # measurements and derived features, with value_origin
profiles = reader.profile_values()  # original marginal counts and log probabilities
```

`measurements()` returns **one row per measurement**, enriched with source and
condition fields. It does not explode measurements over enzymes or cut contexts.
`values()` retains a list of context IDs on each measurement. `contexts()` exposes
P4, P3, P2, P1, P1p, P2p, P3p, P4p and the corresponding position status fields.
`features()` distinguishes `context_source_db` from the MEROPS profile source.
`table(name)` exposes each underlying table without hiding unresolved records.

## Conditions, assays, participants

| Table | Row identity and role |
| --- | --- |
| `source_records` | Database, subset, original ID, raw path, locator, publication IDs, original parser input fields |
| `condition_sets` | Hash of the recorded condition description: medium/species/fraction, pH, temperature, buffer, method, mode |
| `assays` | Source-defined experiment/configuration identity; references a condition description |
| `substrates` | Source-scoped substrate/product identity; sequence, chemistry, residue tokens and reported SMILES |
| `enzymes` | Source-scoped enzyme identity and explicitly supplied identifiers/variants |
| `participants` | Entity, role and amount within one assay |
| `observations` | One measured/reported scalar or limit/range; registered quantity and unit |
| `contexts` | One source-supported cut context within an assay |
| `observation_contexts` | Many-to-many association with explicit attribution; no copying of a numeric value |
| `observation_inputs` | Source inputs to a derived response, e.g. treated and matched control signals |
| `specificity_profiles` | One original MEROPS folder/code membership |
| `profile_cells` | Original counts and log probabilities by profile, position and residue |
| `context_features` | Named derived quantity for one context and one explicitly selected profile |

Cardinality: `condition_sets 1:N assays 1:N participants`; an entity may participate
in many assays. Two experiments with equal recorded conditions reuse the condition
ID, but **remain different assays**. Null fields mean missing descriptions, not
experimental equality. Doses belong to participants and never to the shared condition
ID. qPISA peptide mixtures share an assay; the enzyme dose is not copied to every
peptide as though those were independent reactions. EV6 retains the two experimentally
pooled substrate groups and separates treatment, control and replicate configurations.
Molar participant concentrations use M and mass doses use kg; the original value,
range and unit remain in `reported_amount_json`. Mass and molar amounts are never
converted into one another without molecular mass information. SABIO concentrations
are matched through the reported species key; ambiguous matches remain unresolved.

No cross-source deduplication of measurements is performed. Identical sequences do
not establish identical chemistry or independent experiments. Substrate description
keys and curated half-life representation groups support later, explicit identity
resolution; neither silently averages or removes measurements.

## Common quantities and units

The authoritative registry is `integration/merge/schema.py:QUANTITIES`, also stored
in SQLite as `quantities`. All parser outputs use its exact names and units.

| Quantity | Canonical unit | Interpretation |
| --- | --- | --- |
| `half_life` | s | Reported value, approximation or explicit bounds |
| `remaining_fraction` | fraction | Remaining substrate at the recorded time |
| `functional_activity` | null | Unresolved functional readout retained with original text |
| `kcat` | s^-1 | Turnover number |
| `km` | M | Michaelis constant |
| `kcat_km` | M^-1 s^-1 | Catalytic efficiency |
| `source_k` | null | Coefficient named k by the source, without an invented calibration |
| `relative_activity` | percent_reference | Activity relative to the source reference |
| `substrate_log2_depletion` | log2_ratio | log2 control minus log2 treated |
| `product_log2_enrichment` | log2_ratio | log2 treated minus log2 control |
| `fluorescence_signal` | instrument_au | Original fluorescence readout |
| `mass_spec_signal` | instrument_au | Original mass-spectrometry readout |
| `cleavage_occurrence` | count | One source occurrence, without a rate interpretation |
| `merops_count` | count | Original marginal residue count |
| `merops_log_probability` | ln_probability | Original source matrix entry |
| `merops_probability_sum` | probability_sum | Sum of positional probabilities |
| `merops_log_probability_sum` | ln_probability | Sum of positional log probabilities |
| `merops_probability_product` | probability | Product under independent-position representation |

For observed positions J and the original source matrix L:

- primary score: `merops_probability_sum = sum(exp(L[j, residue[j]]), j in J)`;
- auxiliary log score: `merops_log_probability_sum = sum(L[j, residue[j]], j in J)`;
- auxiliary product: `merops_probability_product = exp(merops_log_probability_sum)`.

The sum is not the joint probability of a window. The product uses an independent
position representation; the source supplies marginals, not joint sequence counts.
No background correction or log-odds is used. `n_observed_positions` and
`n_scored_positions` accompany every score: sums over different coverage are not
automatically comparable. Missing positions are omitted without filling them.
An observed residue absent from the MEROPS alphabet (e.g. Dpa) produces a null score
with `unsupported_residue`; no residue replacement or silent omission occurs.
Absent explicit enzyme mapping gives `no_explicit_profile_mapping`. These contexts
and their empirical measurements are retained.

Profile population 34 is an explicit build default, recorded in metadata and
changeable through `--merops-dataset-index`. Original source log probabilities are
used unchanged, including their original smoothing. Overlapping MEROPS populations
are never added together as independent evidence. Marginals are not reconstructed
into synthetic individual cleavage events.

## Parser mappings

Each dataset has its own `parsers/<name>.py` and the same
`parse(root, *, data=None)` interface. `root` is the acquired series root;
`parse_dataset(name, data_root)` chooses that root and verifies acquired hashes.
Optional `data` reuses an already loaded, dataset-specific source representation.

| Registered parser | Source mapping |
| --- | --- |
| `peplife2`, `pepmsnd`, `dramp`, `plifepred`, `cavaco` | Reviewed half-life adapters; preserve chemistry, limits, approximate values and other readouts |
| `brenda` | All acquired EC 3.4 entries for three kinetic families; mM to M, inverse mM to inverse M |
| `sabio` | All acquired reaction records/participants; selected kinetic parameters use source-normalized SI fields |
| `merops` | All original indexed folders, dictionaries and count/log-probability cells; optional dataset ID selection |
| `mmp_pics` | PICS Output occurrence rows; trypsin/GluC strata remain separate; alternate Unique representation is not counted again |
| `mmp_pics_qf` | Source k, relative activity summaries and time-resolved fluorescence of defined substrates |
| `qpisa_ev1`, `qpisa_ev4` | All signed endpoint responses plus their treated/control inputs; unresolved parent associations remain present |
| `qpisa_ev6` | All source signals and time/replicate-matched substrate/product changes |

BRENDA/SABIO context recovery requires source-supported chemistry: explicit slash or
bond annotation, a recognized terminal reporter bond, or a uniquely supported
substrate/product partition. It does not predict cuts from enzyme preferences.
A kinetic parameter linked to several possible cuts remains **one** measurement;
the association is explicitly marked as not establishing a site-specific rate.

Two descending BRENDA ranges (`0.44-0`, `0.42-0`) remain unresolved with original
text and `unresolved_source_descending_range`; no endpoint reordering is assumed.
SABIO entries without the selected three parameter families still retain their
source records and participants. Source k, relative depletion and activity are not
renamed kcat. qPISA geometric windows are labeled by evidence and are not all
asserted to be independently demonstrated cuts.

## Build, provenance and extension

```bash
.venv/bin/python assets/scripts/integration/proteolysis_data_manager/merged_data/build_dataset.py --score-windows
# Explicit subset:
.venv/bin/python assets/scripts/integration/proteolysis_data_manager/merged_data/build_dataset.py \
  --source brenda --source sabio --output /tmp/kinetics.sqlite
```

The public script is reached through the existing library directory shortcut;
conversion logic exists only in the library. Output is one atomic SQLite file.
The reader checks schema version, quantity definitions, foreign keys, units, scope
and positional semantics. Build metadata (timestamp, source manifest hashes, parser
implementation hashes, selected datasets and score population) live inside SQLite.
No new raw acquisition receipts or integration JSON/CSV files are created.

Raw inputs and their one existing acquisition manifest per series remain unchanged.
Every empirical value points to its source; derived scores additionally point to
the original profile. The original source record input is retained as JSON in the
database, including fields not selected into canonical columns.

To add a dataset, add one parser module and registry entry; emit the existing entity,
condition, observation and context tables. Add a quantity to the central registry
only when it represents a genuinely new measured quantity, with explicit units and
derivation. Do not encode a database-specific name for an existing quantity. Schema
changes require a schema version change and explicit reader compatibility handling.
