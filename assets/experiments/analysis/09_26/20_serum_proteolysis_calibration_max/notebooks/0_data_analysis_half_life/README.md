# Half-life data analysis

This series handles half-life data. MEROPS/cut windows and kinetic datasets
are separate work.

Implementation is in the independent
[`proteolysis_data_manager` library](../../../../../../tools/proteolysis_data_manager/README.md).
Notebooks import its acquisition, integration, protocol and analysis modules;
source adapters live under `integration/half_life/adapters/`.

## Source overviews

| Notebook | Scope |
|---|---|
| `0_01_peplife2_overview` | Source size, documentation/missingness, label semantics/classes and species/matrix groups. |
| `0_01a_peplife_publication_subsets` | Published PlifePred subsets versus current PEPlife2: overlap, changed labels and added/missing information. |
| `0_02_dramp_overview` | DRAMP measurement types, human/mouse serum/plasma coverage and documentation. |
| `0_02a_pepmsnd_overview` | Separate published PepMSND dataset audit. |
| `0_02b_cavaco_baseline_overview` | Experimental Cavaco reference: 16 printed validation points and reported CI. |
| `0_02c_dramp_dilution_pairs` | Matched explicit dilution measurements; exploratory linear-intensity hypothesis, not a protocol transformation. |

Overviews reconstruct detailed source records from verified raw files in memory.
This supports raw wording and publication audits without saving a second wide
normalized table. Non-exact label decisions, historical discrepancies and shared
field enrichment are listed in `data/half_time_datasets/normalized/remarks_<source>.md`.
Cavaco serves as the shared baseline; sequence overlap and chemical/environmental
agreement are distinguished.

## Condition protocols and aggregation

`protocols/` contains `protocol_base.ipynb`, `aggregation_base.ipynb`,
`protocol_config.json` and [CLEAN_SCHEMA.md](protocols/CLEAN_SCHEMA.md).
Each condition has its own folder: `human_serum`, `human_plasma`, `mouse_serum`,
`mouse_plasma`. Each contains one source protocol per database and a separate
`aggregation_<condition>.ipynb`. Templates have exports disabled.

Run the download/normalization scripts first, then source protocols, then that
condition's aggregation. Protocols load the compact normalized Parquet and
select species/matrix/mode. Only pH and temperature gate acceptance. Missing
values use reference assumptions transiently; exported reported fields stay null.
Matrix fraction, concentration and C/f are retained for later modeling.

Each acceptance parameter has one plot showing reported versus assumed values,
the reference baseline and acceptance interval, followed by explicit rejection
counts. Duration classes have consistent colors: short <30 min (blue), medium
30 min–6 h inclusive (orange), long >6 h (green), unresolved (gray). Time axes
use seconds/minutes/hours/days rather than log10 labels.

## Data contract and files

```text
data/half_time_datasets/
  raw/<source>/
  normalized/observations.parquet
  normalized/remarks_<source>.md
  processed/<condition>/sources/<source>.csv
  processed/<condition>/<condition>.csv
  manifests/
  views/                       # preceding compatibility outputs, untouched
```

Normalized Parquet, protocol CSVs and final CSVs have the **same 23 columns**.
Normalization consolidates verified shared measurements across all conditions,
without acceptance filtering. Compatible missing fields are enriched with donor
IDs documented in remarks. Source lists and all record IDs survive every stage.
Known conflicts remain separate; equal sequences alone never justify merging.

CSV source/reference lists are JSON. Blank scalar cells are null. Read using
`proteolysis_data_manager.integration.half_life.schema.read_csv`. Each accepted numerical,
class-only and unresolved observation survives into aggregation. Source accounting
shows raw representations, normalized measurements, mode/gate reductions,
label availability, missingness and sequence/verified-measurement overlap with
PEPlife2. Diagnostics remain inside notebooks, without report files.

Only reported numerical points populate `half_life_seconds`, except an explicitly
documented CI midpoint with no reported point. Ordinary ranges, censoring and
approximation are not numerical points. Missing CI/SD remain null; an untyped ±
is documented in remarks. A later uncertainty estimator is a protocol TODO.
Full field semantics are in [CLEAN_SCHEMA.md](protocols/CLEAN_SCHEMA.md), with
source mapping details in the downloader's
[DATA_DICTIONARY.md](../../../../../../scripts/downloads/half_time_data/DATA_DICTIONARY.md).
