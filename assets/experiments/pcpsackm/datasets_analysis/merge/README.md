# Common proteolysis dataset

- `0_01_common_parameters.ipynb`: load the SQLite backend, compute the shared numeric
  feature table `merged`, display overall source statistics and actual values.
- `0_02_position_probabilities.ipynb`: normalize MEROPS/PICS occurrences and weighted
  qPISA candidate windows to the same positional amino-acid probability matrix.

The primary interface is `dataset.to_frame()`. Kinetic constants, relative substrate
loss, QF activity and positional scores are numeric columns with named reference
bases and source provenance. Both notebooks operate in memory and write no CSV.

Implementation and formulas:
`assets/tools/proteolysis_data_manager/docs/UNIFIED_CONTRACT.md`.
