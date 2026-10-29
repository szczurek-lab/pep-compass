# Documentation

Documentation of the proteolysis data manager. The project entry point is [../README.md](../README.md).

## Contents

### Acquisition

- [Raw acquisition](RAW_ACQUISITION.md) — shared commands, receipts, validation and source inventory.
- [Half-life workflow](HALF_LIFE_WORKFLOW.md) — building the two-table half-life dataset and the Python API.
- [Kinetics workflow](KINETICS_WORKFLOW.md) — BRENDA and SABIO-RK acquisition, BRENDA normalization, outputs.
- [Window-neighbourhood workflow](WINDOW_NEIGHBOURHOOD_WORKFLOW.md) — MEROPS, MMP/PICS and DPP4/qPISA acquisition and the MEROPS CSV.

### Data contracts

- [Half-life data dictionary](HALF_LIFE_DATA_DICTIONARY.md) — fields of the two half-life tables.
- [Kinetics contract](KINETICS_CONTRACT.md) — eight-table bundle, keys, units, reader, protocol filter.
- [Unified contract](UNIFIED_CONTRACT.md) — common SQLite backend, `to_frame()` columns, parsers, public API.

### Validation reports

- [Unified contract validation](UNIFIED_CONTRACT_VALIDATION.md) — executed counts and checks of the SQLite backend.
- [Wide parameters validation](WIDE_PARAMETERS_VALIDATION.md) — executed checks of the common wide table.

### Library

- [Library structure](LIBRARY_STRUCTURE.md) — packages and their responsibilities.
- [Launcher scripts](../scripts/README.md) — canonical scripts and repository shortcuts.

### Analysis notebooks

Notebooks that display these datasets are indexed in
[assets/experiments/pcpsackm](../../../experiments/pcpsackm/README.md).
