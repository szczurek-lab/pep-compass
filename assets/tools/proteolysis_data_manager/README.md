# Proteolysis data manager

Independent library for acquiring, integrating, filtering and inspecting
proteolysis datasets. It does not import the peptide model or Torch. Half-life
and the initial BRENDA kinetics workflow are implemented as separate families;
cleavage-window integration and cross-family enrichment are deferred.

The [kinetics contract and relationship diagram](docs/KINETICS_CONTRACT.md)
describe the eight-table bundle and its reader. Acquisition commands are in the
[kinetics workflow](../../scripts/downloads/kinetics_data/README.md).

## Install and run

```bash
python -m pip install -e assets/tools/proteolysis_data_manager
proteolysis-download-half-life --source peplife2
proteolysis-normalize-half-life
```

For kinetics, import an already acquired BRENDA release and normalize all EC 3.4
entries, including non-human organisms:

```bash
proteolysis-import-brenda --archive PATH_TO_ARCHIVE --receipt PATH_TO_ORIGINAL_RECEIPT
proteolysis-normalize-kinetics
```

Half-life defaults use `data/half_time_datasets`; kinetics defaults use
`data/kinetics_datasets` in this repository. Pass `--data-root PATH`
to work with a different snapshot, including outside the repository.
The existing scripts in `assets/scripts/downloads/half_time_data` are thin
entry points to this library and also work directly from a checkout.
Notebook bootstraps add this package's `src` directory, so they do not require
installing the model package.

## Module responsibilities

```text
src/proteolysis_data_manager/
  acquisition/
    half_life.py                 # source URLs, retrieval, raw verification
    kinetics/brenda.py            # exact local snapshot import and verification
    resources/                   # receipts from the preceding source audit
  integration/
    half_life/
      adapters/                  # source-specific readers and curated resources
      normalization.py           # raw inventory -> common normalized Parquet
      schema.py                  # columns, types, units, identity and I/O
      labels.py                  # numerical/class label interpretation
      remarks.py                 # concrete decisions with source identifiers
      aggregation.py             # merge accepted source tables per condition
    kinetics/
      adapters/brenda.py          # all-organism source entries and associations
      schema.py                  # eight typed tables, units and primary keys
      normalization.py           # source archive -> complete relational bundle
      reader.py                  # validation, closed subsets and analysis view
  protocols/
    half_life/
      conditions.py              # condition keys, assumptions and gate decisions
      filtering.py               # protocol input, accepted rows and exports
    kinetics/filtering.py         # one reviewed cohort, no matrix assumptions
  analysis/
    half_life/
      overview.py                # completeness, source overlap and summaries
      visualization.py           # plots and shared display conventions
    kinetics/overview.py          # association counts and per-unit distributions
  utils/                         # paths, manifests and file helpers
  cli.py                         # installed command entry points
```

Adapters belong to integration: they map the acquired source files into the
same observation vocabulary. Acquisition does not normalize measurements;
normalization does not apply acceptance gates. Protocols select one organism
and matrix. Aggregation consolidates the accepted measurements for that
condition, preserving contributor identifiers and both numeric and class-only
labels. It does not merge records by sequence alone.

## Python usage

```python
from pathlib import Path
from proteolysis_data_manager.acquisition.half_life import acquire
from proteolysis_data_manager.integration.half_life.normalization import build_normalized_dataset
from proteolysis_data_manager.integration.half_life.schema import read_csv

root = Path("data/half_time_datasets")
acquire("peplife2", root)
# Build after acquiring all five supported sources.
manifest = build_normalized_dataset(root)
accepted = read_csv(root / "processed/human_serum/human_serum.csv")
```

The half-life contract contains 23 fields. Source/reference lists are native
Parquet lists and JSON in CSV; null scalar values remain null. Reported data,
class thresholds and acceptance policies are unchanged by the package move.
See the [download workflow](../../scripts/downloads/half_time_data/README.md),
[field dictionary](../../scripts/downloads/half_time_data/DATA_DICTIONARY.md)
and [notebook series](../../experiments/analysis/09_26/20_serum_proteolysis_calibration_max/notebooks/0_data_analysis_half_life/README.md).

## Verification

From the repository root, with the library dependencies and pytest installed:

```bash
python -m pytest tests/half_life tests/kinetics tests/assets/scripts/downloads/serum_proteolysis_calibration -q
```

The condition notebooks exercise the source protocol and final aggregation on
the local raw snapshot. Base notebooks remain unexecuted templates.
