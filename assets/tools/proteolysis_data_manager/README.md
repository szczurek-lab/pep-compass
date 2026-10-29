# Proteolysis data manager

Library for acquiring, integrating, filtering and inspecting proteolysis datasets. It does not
import the peptide model or Torch.

## Introduction

The library handles three raw-data series:

- half-life: PEPlife2, PepMSND, PlifePred, DRAMP, Cavaco;
- window neighbourhoods: MEROPS, MMP/PICS, DPP4/qPISA;
- kinetics: BRENDA, SABIO-RK.

Integration covers half-life, BRENDA, SABIO-RK, the original MEROPS profiles, MMP/PICS and
DPP4/qPISA. All integrated sources are stored in one versioned SQLite contract with a shared reader.

### Main capabilities

- Acquisition of unchanged source files with checksummed receipts.
- Half-life integration into two TSV tables.
- Kinetics integration into an eight-table relational bundle (BRENDA).
- Unified SQLite backend; the common numeric table is `ProteolysisDataset.read(path).to_frame()`.

## Installation

```bash
python -m pip install -e assets/tools/proteolysis_data_manager
```

Notebooks add this package's `src` directory to the path and do not require installing the model package.

## Quick start

From the repository root:

```bash
# 1. Acquire raw data (all three series)
.venv/bin/python assets/scripts/downloads/proteolysis_data_manager/download_all.py

# 2. Build the unified SQLite backend
.venv/bin/python assets/scripts/integration/proteolysis_data_manager/merged_data/build_dataset.py --score-windows
```

The backend is written to `data/proteolysis/merged_datasets/proteolysis.sqlite`. A first BRENDA
import requires a licensed archive; see the [kinetics workflow](docs/KINETICS_WORKFLOW.md#brenda).

```python
from proteolysis_data_manager.integration.merge import ProteolysisDataset

merged = ProteolysisDataset.read("data/proteolysis/merged_datasets/proteolysis.sqlite").to_frame()
```

Default data roots are `data/proteolysis/<series>`, where `<series>` is `half_time_datasets`,
`kinetics_datasets` or `window_neighbourhood_datasets`. Per-series commands and options are in the
[documentation](docs/README.md).

## Documentation

The full documentation index is [docs/README.md](docs/README.md). Principal sections:

- Acquisition: [raw acquisition](docs/RAW_ACQUISITION.md) and the half-life, kinetics and
  window-neighbourhood workflows.
- Data contracts: half-life data dictionary, kinetics contract, unified contract.
- Validation reports.
- [Library structure](docs/LIBRARY_STRUCTURE.md) and [launcher scripts](scripts/README.md).

Analysis notebooks are indexed in [assets/experiments/pcpsackm](../../experiments/pcpsackm/README.md).

## Verification

From the repository root, with the library dependencies and pytest installed:

```bash
python -m pytest tests/half_life tests/kinetics tests/acquisition -q
```
