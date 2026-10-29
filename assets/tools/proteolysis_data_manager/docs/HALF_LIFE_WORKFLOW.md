# Half-time integration workflow

The maintained command builds two human-readable datasets from the acquired raw
snapshot. It performs no network acquisition or assay acceptance filtering.

```bash
.venv/bin/python assets/tools/proteolysis_data_manager/scripts/integration/half_time_data/normalize_datasets.py
```

The repository shortcut is the same file through a directory symlink:

```bash
.venv/bin/python assets/scripts/integration/proteolysis_data_manager/half_time_data/normalize_datasets.py \
  --data-root data/proteolysis/half_time_datasets
```

`--data-root PATH` selects a different input series. `--output-dir PATH` overrides
the default `DATA_ROOT/processed` destination. Defaults are declared at the top
of the canonical launcher. Existing raw receipts and files are verified before
parsing, and no download occurs.

```text
data/proteolysis/half_time_datasets/
  raw/<source>/                      # unchanged published artifacts
  raw/manifest.json                  # acquisition receipt
  processed/sequences_half_time.tsv  # peptide descriptions, seven fields
  processed/half_time_experiments.tsv # source observations, 25 fields
```

Both outputs are staged before replacement. Other existing outputs are not
silently deleted. The provisional Parquet pair produced during development was
moved to `tmp/archive_old_analysis/half_time_two_table_migration/provisional_outputs`.

The five source parsers remain the existing database-specific implementations.
`adapters/common.py` owns unit/condition parsing; `adapters/schema.py` owns the
shared contract and split; `datasets.py` owns the single source orchestrator,
verified cross-source context recovery and TSV I/O. It reads PEPlife2 once and
passes its inventory to PlifePred's existing author-ID link logic. The reviewed
DRAMP context matching rule has one implementation in this orchestrator.

Cavaco uses the existing reviewed 16-peptide Figure 3a transcription. Its PDF
checksum and XML Table 1 sequences are checked. This does not reconstruct the
129-peptide training dataset or digitize degradation curves.

The analysis notebooks `0_02_processed_dataset.ipynb` and
`0_03_estimation_interval_analysis.ipynb` under
[`datasets_analysis/half_time`](../../../experiments/pcpsackm/README.md) read only the processed
TSVs. Their calculations live in `analysis/half_life/processed.py` and
`analysis/half_life/intervals.py`; figures reuse the existing visualization module.
Derived tables, model coefficients and standalone figures are stored beside each
notebook in its `_results/` directory. These results are reproducible from the
processed input pair and are not a second integration export.

Reported half-life bounds and approximations stay explicit. Only exact positive
half-lives enter the distribution/agreement plots. Equal missing numeric
conditions mean equal recorded descriptions, not established experimental
identity. Empirical CI-shape estimation uses peptide-held-out validation and is
restricted to the documented Cavaco assay design; it does not validate CI coverage.

Historical versions, reviewed exceptions and migration notes are retained in
`tmp/archive_old_analysis/half_time_two_table_migration`. Prior flat protocol
helpers remain available for archived analysis consumers, but the active builder
and both command entry points write only the new TSV pair.

## Python API

```python
from pathlib import Path
from proteolysis_data_manager.integration.half_life.datasets import build_half_time_datasets, read_half_time_datasets

root = Path("data/proteolysis/half_time_datasets")
dataset = build_half_time_datasets(root)  # uses all five acquired sources
loaded = read_half_time_datasets(root / "processed")
sequences = loaded.sequences_half_time
experiments = loaded.half_time_experiments
joined = experiments.merge(sequences, on="sequence_id", validate="many_to_one")
```

The contract has seven peptide fields and 25 experiment fields, stored as UTF-8
`processed/sequences_half_time.tsv` and `processed/half_time_experiments.tsv`; see the
[data dictionary](HALF_LIFE_DATA_DICTIONARY.md). Every source observation is retained, including
repeated author views. No descriptors, SMILES, sequence lengths, inferred conditions or CI midpoint
labels are generated.

The `proteolysis-normalize-half-life` command calls the same builder; `build_normalized_dataset`
delegates to it. `normalization.load_source_records` reconstructs the detailed compatibility view
used by the source overviews; it reuses the same source reader and writes no dataset export.
