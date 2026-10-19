# Kinetics data analysis

This series starts with BRENDA, preserving all organisms in normalized data.
It does not locate cleavage fragments, match MEROPS, enrich substrate chemistry
from other families or run protocols for additional sources.

| Notebook | Purpose |
|---|---|
| `0_01_brenda_overview.ipynb` | Eight-table inventory, organism associations, missingness, substrate descriptions, parameter/condition distributions and source references. |
| `0_01a_brenda_source_context_audit.ipynb` | Raw entry wording, unresolved protein links, ranges, units and free-text mentions; no automated site localization. |
| `protocols/human_enzyme/protocol_human_enzyme_brenda.ipynb` | First review protocol: exclusively human enzyme associations, explicit gate configuration and complete relational export. |
| `protocols/protocol_base.ipynb` | Unexecuted copyable template; exports disabled. |

Run [acquisition and normalization](../../../../../../scripts/downloads/kinetics_data/README.md)
first, then the overview/audit, then the first protocol. Import paths point to the
independent [data-manager library](../../../../../../tools/proteolysis_data_manager/README.md).
The [contract documentation](../../../../../../tools/proteolysis_data_manager/docs/KINETICS_CONTRACT.md)
includes the saved relationship graph, exact columns, units and missingness rules.

## Scope of the first protocol

`human_enzyme` identifies source contexts whose linked enzyme species are all
Homo sapiens, with no unknown linked enzyme species. It does **not** identify
human serum, human plasma, or the organism from which the assay sample came.
This name refers only to the species of the enzyme associations.

The current configuration leaves experiment `species`, `matrix`, and
`in_vivo_in_vitro` selectors unset and disables pH/temperature acceptance
bounds. It therefore retains every measurement in accepted enzyme contexts,
including measurements with missing conditions and nonnumeric values. The
post-filter output is the distribution of BRENDA source records associated with
contexts that have exclusively documented human enzyme links. It is not a
serum-specific distribution or a count of independent experiments. Sample
matrix and species are reported only when curated in the source; they are not
inferred from enzyme organism or tissue. Several linked human proteins remain
associations, not independently measured enzyme-specific parameters.

The pH 7.4 and 37°C markers are display references. They never fill missing
values. pH/T acceptance bounds are disabled in the first configuration, ready
for review; changing them requires an explicit missing-value policy. Missing
sequences, cleavage annotations and numerical labels are not export gates.
Positive point eligibility for later numerical modeling is reported separately.
No SD/CI is synthesized.

## Files

```text
data/kinetics_datasets/
  raw/brenda/
  normalized/{enzymes,substrates,experiments,...}.parquet
  normalized/remarks_brenda.md
  processed/human_enzyme/sources/brenda/<same eight tables>.parquet
  manifests/
```

The reader loads the tables as `KineticsDataset`, validates their references and
returns closed relational subsets. A flattened analysis view has one row per
measurement with list-valued enzyme/site associations. Separate cleavage sites
never multiply a global kinetic measurement. No redundant CSV, report folders
or duplicate exported figure files are generated; plots are saved in notebook outputs.

Use the explicit table IDs as join keys: `experiment_id`, `enzyme_id`,
`substrate_id`, `measurement_id`, and `site_id`. The two `experiment_*_relation`
tables connect experiments to enzymes/substrates; the measurement/site relation
connects a measurement to explicitly reported sites. Joining all these tables
before aggregating can duplicate measurements. The full primary/foreign-key
map and the exact filter interpretation are documented in the
[kinetics contract](../../../../../../tools/proteolysis_data_manager/docs/KINETICS_CONTRACT.md#primary-and-join-keys).

Experiment counts are source-context counts, not established independent
experiments or replicates. Species-associated counts can overlap when a source
entry links several organisms. Future datasets and a final aggregate await review
of this first protocol. Half-life and cleavage-window series remain separate.
