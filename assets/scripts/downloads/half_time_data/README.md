# Peptide half-life datasets

This directory contains thin command entry points for peptide half-life sources.
The implementation lives in the independent
[proteolysis_data_manager library](../../../tools/proteolysis_data_manager/README.md),
organized into acquisition, integration/adapters, protocols, analysis and utils.
Download receipts and all pipeline manifests are centralized in `data/half_time_datasets/manifests/`. Existing legacy receipts are migrated without changing retrieval metadata.
All current artifacts belong in `data/half_time_datasets`. Source overviews inspect detailed records reconstructed from raw files in memory. Source protocol notebooks
write reviewed environment-specific processed tables; they do not download data.
The preceding source investigation is preserved in [AUDIT.md](AUDIT.md).
For a field-by-field explanation of source mappings, calculated descriptors,
analysis flags and missing-value semantics, see [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Run

From the repository root, using the project Python environment:

```bash
uv run assets/scripts/downloads/half_time_data/download_datasets.py
uv run assets/scripts/downloads/half_time_data/normalize_datasets.py
```

The acquisition script uses the standard library. Normalization uses the
project's pandas, pyarrow, openpyxl and Bio dependencies; it does not import
Torch or require model weights. Both scripts also accept `--data-root PATH`.

A source can be acquired separately with `--source peplife2`, `pepmsnd`,
`plifepred`, `dramp` or `cavaco`. `--force` explicitly replaces current raw
artifacts; reruns without it retain verified files and original retrieval times.
Do not manually edit raw files or source manifests.

To reuse the existing DRAMP snapshot while preserving its original receipt:

```bash
uv run assets/scripts/downloads/half_time_data/download_datasets.py \
  --source dramp --import-legacy data/serum_proteolysis_calibration
```

`--import-audit PATH` imports named artifacts from a preceding local audit.
Imports are marked as imports, with checksums and import time; unknown original
retrieval time stays unknown. They are never reported as new network downloads.

## Sources

| Source | Acquisition and scope |
| --- | --- |
| PEPlife2 | Natural + Modified API responses; full union of IDs checked against complete FASTA. Preserve all species, matrices and readouts. |
| PepMSND | Published XLSX pinned to Git commit `c0fa6957c7f88a57af29410cfc075542c92a2c63`; preserve hours, SMILES, chemistry and environment. |
| PlifePred | Published natural and mixed XLSX supplements; keep author datasets separately. Natural is a subset of mixed. |
| DRAMP | Original `stability_amps.xlsx`; preserve half-life, fraction-at-time and functional-activity text as distinct readouts. |
| Cavaco | Versioned PMC article and supplements, plus 16 reviewed experimental human-serum validation records. |

Cavaco numerical half-lives and 95% CI are transcribed from **printed Figure 3a
labels**, not estimated from curve coordinates. The checked-in
[provenance](../../../tools/proteolysis_data_manager/src/proteolysis_data_manager/integration/half_life/adapters/resources/cavaco_validation.provenance.json) documents the extraction.
The normalizer checks the source PDF checksum, Table 1 sequences and interval
consistency. The 129-peptide regression training set is not claimed as
reconstructed. The large RAR contains an Electron application; its application
resources were inspected during development and were not found to provide a
standalone half-life dataset. No application executable is run or downloaded by
the production pipeline.

## Outputs and shared contract

```text
data/half_time_datasets/
  raw/<source>/
  normalized/observations.parquet     # 23-field contract, all conditions
  normalized/remarks_<source>.md      # concrete label/enrichment decisions
  processed/<condition>/sources/<source>.csv
  processed/<condition>/<condition>.csv
  manifests/                         # all receipts/checksums
  views/                             # existing compatibility data, untouched
```

The normalizer calls the source adapters in
`proteolysis_data_manager.integration.half_life.adapters`, consolidates verified aliases
and fills compatible missing fields from their contributors. `source` retains
all databases; `source_record_id` retains all view/record/readout references.
Known conflicts remain separate. Original non-exact/error wording and concrete
decisions remain in `raw/` and `remarks_<source>.md`. Sequence equality alone
does not establish measurement identity.

Normalized Parquet, source CSVs and final condition CSVs use the **same 23 fields**,
listed in [CLEAN_SCHEMA.md](../../../experiments/analysis/09_26/20_serum_proteolysis_calibration_max/notebooks/0_data_analysis_half_life/protocols/CLEAN_SCHEMA.md).
Parquet stores lists natively; CSV encodes source/reference lists as JSON and
null scalars as blank cells. Use the library `schema.read_csv` to load CSV.
No extra report files, subset exports or wide provenance columns are generated.

## Notebooks and filtering

English notebooks are in
[0_data_analysis_half_life](../../../experiments/analysis/09_26/20_serum_proteolysis_calibration_max/notebooks/0_data_analysis_half_life/README.md).
Overviews rebuild detailed source records in memory for documentation, chemistry,
source comparison and raw wording audits. They do not apply the source protocol.
The PEPlife2 publication subsets, DRAMP, PepMSND, Cavaco and paired dilution
analysis remain separate notebooks.

Source protocols read the compact normalized Parquet. Each condition has its
own source notebooks and `aggregation_<condition>` notebook, for human/mouse
serum/plasma. Missing pH/T use approved references only for the decision, while
exported reported values remain null. Dilution and concentration are retained
for later modeling; neither is an acceptance gate.

Every accepted numerical, class-only and unresolved observation survives into
the source CSV and final aggregate. Missing values remain null. Project classes:
short <30 min, medium 30 min–6 h inclusive, long >6 h. Ordinary ranges and
censoring never become numerical midpoint labels. A documented CI without a
reported point may supply a representative midpoint, explicitly noted in remarks.
Missing CI/SD are never invented, and untyped ± stays untyped. The protocol's
uncertainty estimator remains a TODO for later analysis.

`views/` remains for earlier calibration; this normalizer does not update it.
Calibration, cut windows/MEROPS and kinetic data are outside this stage.

The serum acquisition orchestrator routes its three legacy half-life source keys
here. Compatibility download entrypoints retain explicit old output overrides;
default invocations use this canonical location. Existing legacy files remain
available for old snapshots but are not the inputs to updated notebooks.

Grouping for future evaluation connects identical sequences and shared
publications across sources. No folds or predictive scores are claimed;
similarity clustering and the model's supported chemistry must be decided before
reporting generalization to unseen peptide families.

## Verification

```bash
uv run pytest tests/half_life/test_half_life.py \
  tests/assets/scripts/downloads/serum_proteolysis_calibration/test_downloaders.py
```

The tests check label censoring, duration-class ambiguity, reported untyped
errors, split measurements, source provenance, FASTA completeness, receipt
retention, failed-download atomicity, evaluation grouping and protocol exports.
Source-overview notebooks and the PEPlife2 protocol are executed on the acquired
snapshot. The base protocol remains an unexecuted copyable template.
