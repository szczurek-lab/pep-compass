# Library structure

Package layout and the responsibility of each stage of `src/proteolysis_data_manager`.

## Scope

Data contracts are defined in [UNIFIED_CONTRACT.md](UNIFIED_CONTRACT.md) and
[KINETICS_CONTRACT.md](KINETICS_CONTRACT.md). Acquisition rules are in
[RAW_ACQUISITION.md](RAW_ACQUISITION.md). Launcher scripts are listed in
[../scripts/README.md](../scripts/README.md).

## Stages

| Stage | Package | Responsibility |
|---|---|---|
| Acquisition | `acquisition/` | Stores unchanged source files and validates them |
| Integration | `integration/` | Maps acquired files to a common observation vocabulary |
| Protocols | `protocols/` | Selects one organism and matrix; applies acceptance gates |
| Analysis | `analysis/` | Overviews and plots for notebooks |

Adapters belong to integration. Acquisition does not normalize measurements; normalization does not
apply acceptance gates. Aggregation consolidates the accepted measurements of one condition,
preserving contributor identifiers and both numeric and class-only labels. It never merges records
by sequence alone.

## Package tree

```text
src/proteolysis_data_manager/
  acquisition/
    download_all.py              # orchestrate the independent data series
    helpers/
      raw.py                     # unchanged bytes, staging, shared raw receipt
    half_life/
      download_all.py            # destinations and source orchestration
      peplife2.py                # acquire + source validation
      pepmsnd.py
      plifepred.py
      dramp.py
      cavaco.py
    kinetics/
      download_all.py            # kinetics source orchestration
      brenda.py                  # licensed original archive import + validation
      sabio.py                   # complete official Export API JSON pages
    window_neighbourhood/
      download_all.py            # destinations and source orchestration
      merops.py                  # unchanged original curated dictionary tree
      mmp_pics.py                # original publication artifacts
      dpp4_qpisa.py              # original publication artifacts
  integration/
    half_life/
      adapters/                  # source readers, common parsing, two-table schema
      datasets.py                # source orchestration, peptide/experiment TSV I/O
      normalization.py           # analysis views use the same source loader
      schema.py                  # legacy flat protocol contract
      labels.py                  # numerical/class label interpretation
      remarks.py                 # concrete decisions with source identifiers
      aggregation.py             # merge accepted source tables per condition
    kinetics/
      adapters/brenda.py         # all-organism source entries and associations
      schema.py                  # eight typed tables, units and primary keys
      normalization.py           # source archive -> complete relational bundle
      reader.py                  # validation, closed subsets and analysis view
    window_neighbourhood/
      merops.py                  # original matrix cells and dictionary identities -> CSV
    merge/                       # unified SQLite backend; see UNIFIED_CONTRACT.md
  protocols/
    half_life/
      conditions.py              # condition keys, assumptions and gate decisions
      filtering.py               # protocol input, accepted rows and exports
    kinetics/filtering.py        # one reviewed cohort, no matrix assumptions
  analysis/
    half_life/
      overview.py                # completeness, source overlap and summaries
      visualization.py           # plots and shared display conventions
      processed.py               # calculations of the processed-dataset notebook
      intervals.py               # calculations of the estimation-interval notebook
    kinetics/overview.py         # association counts and per-unit distributions
    window_neighbourhood/        # calculations of the window-neighbourhood notebooks
  utils/                         # paths, manifests and file helpers
  cli.py                         # installed command entry points
```

The internal layout of `integration/merge/` is described in the
[module structure](UNIFIED_CONTRACT.md#module-structure) section of the unified contract.
