# Canonical proteolysis launchers

These scripts invoke the library CLI directly. Source acquisition, validation and
integration implementations live in `src/proteolysis_data_manager`.

The launchers define their default series roots at the top of each file; explicit CLI arguments take
precedence. Destinations, receipts, options and rerun behaviour are described in
[raw acquisition](../docs/RAW_ACQUISITION.md).

For example, set a half-life destination directly from the script:

```bash
python assets/scripts/downloads/proteolysis_data_manager/half_time_data/download_all.py \
  --data-root data/proteolysis/half_time_datasets \
  --source peplife2
```

- `downloads/download_all.py`: acquire the selected raw-data series.
- `downloads/half_time_data/download_all.py`: acquire all or selected half-life databases.
- `downloads/half_time_data/download_<source>.py`: select one database through the same CLI.
- `downloads/window_neighbourhood_data/download_all.py`: acquire MEROPS, MMP/PICS and DPP4/qPISA; repeat `--source` to select sources.
- `downloads/window_neighbourhood_data/download_merops.py`: copy the unchanged original `data/merops` tree; `--merops-source PATH` selects another original directory.
- `downloads/window_neighbourhood_data/download_<source>.py`: acquire one window source.
- `downloads/kinetics_data/download_all.py`: reuse/import BRENDA and acquire SABIO-RK; repeat `--source` to select sources.
- `downloads/kinetics_data/download_sabio.py`: acquire original peptidase JSON pages; `--sabio-query` changes the explicit source query.
- `downloads/kinetics_data/import_brenda.py`: register a licensed original BRENDA archive.
- `integration/half_time_data/normalize_datasets.py`: build the peptide and experiment TSV pair in `DATA_ROOT/processed`; `--output-dir` overrides the destination.
- `integration/kinetics_data/normalize_datasets.py`: invoke the existing kinetics normalization workflow.
- `integration/merged_data/build_dataset.py`: build the unified SQLite backend (`--score-windows`); see the [unified contract](../docs/UNIFIED_CONTRACT.md).
- `integration/window_neighbourhood_data/merge_merops.py`: export original MEROPS matrix cells and dictionary identities to `processed/merops_matrices.csv`; `--output-path` overrides the CSV destination.

Repository shortcuts are directory symlinks:
`assets/scripts/downloads/proteolysis_data_manager` and
`assets/scripts/integration/proteolysis_data_manager`.
Paths are resolved from the canonical file, so launchers also work from another
working directory. Relative data-path arguments are still relative to that
working directory.

Use `--help` for arguments. The integration contract remains subject to the
separate review; moving its launchers does not establish new scientific validation.
