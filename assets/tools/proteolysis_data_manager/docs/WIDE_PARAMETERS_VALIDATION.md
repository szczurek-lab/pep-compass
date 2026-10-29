# Common wide parameters: validation

The primary table is `ProteolysisDataset.read(path).to_frame()`. Scientific
transforms are implemented in `features/activity.py` and `features/frequencies.py`;
source-scoped association and pivoting are in `views/wide.py`. The normalized
SQLite format remains compatible with schema 1.0. This revision did not rebuild
or modify the existing SQLite database or acquired raw files.

## Executed data checks

The full existing backend produced **229,925 wide rows and 235 columns**. All
**233,021 original measurement IDs**, values and source-record associations remain
recoverable through per-quantity provenance columns. Rows associated with multiple
contexts retain the original measurement IDs. Safe same-source parameter grouping
reduces row count; it does not average measurements.

| Acceptance criterion | Result | Evidence |
| --- | --- | --- |
| Named common numeric columns | PASS | Full frame contains kcat, km, kcat_km, percentages, rates and positional probabilities; no quantity/value columns |
| Original values and source identity preserved | PASS | Every measurement ID, original numeric point and source ID compared with the backend |
| Source-scoped parameter association | PASS | Repeated/unresolved groups remain unpaired; 249 additional efficiencies equal kcat/km from a unique same-source substrate group |
| Percentage transformation | PASS | Checked every qPISA substrate percentage against 100 × (1 − 2^(-d)); remaining + depleted = 100 within 1e-10 percentage points |
| Signed net changes retained | PASS | Negative depletion remains present; no clipping of measured responses |
| Apparent-rate transformation | PASS | Every endpoint rate checked against ln(2) × d/time; intrinsic kcat remains separate |
| Shared positional distributions | PASS | Every positive-depth profile sums to 1 within absolute 1e-14; zero-depth positions remain null |
| PICS occurrence probabilities | PASS | Independently reconstructed all enzyme/library/position distributions from original occurrence/context links |
| Probability-score coverage | PASS | Every sum equals its available p_P4..p_P4p terms; coverage matches nonmissing terms |
| Enzymes lacking MEROPS IDs | PASS | All qPISA endpoint windows, including DPF-3, receive their own source-specific probability profile |
| Source QF normalization | PASS | All 162 round percentages checked against 100 × k/k_PLG; all 54 merged source summaries retained |
| Fluorescence fitting | PASS | 162 slopes available over the source-selected intervals; separate least-squares reconstruction recorded in `/tmp/merge_notebook_final.log` |
| Module organization | PASS | Implementation grouped into contract, parser families, features, views and storage; root public imports preserved |
| Original notebook content and raw files | PASS | All original untagged cells match the pre-change snapshot; all raw checksums match original acquisition manifests |

Common source-specific numerical availability includes:

- qPISA: **134,094** relative substrate-depletion percentages.
- DRAMP: **19** additional point depletion percentages from reported remaining fractions.
- QF: **216** relative activities, consisting of 162 round values and 54 source
  merged summaries; **162** independently refitted fluorescence slopes.
- SABIO-RK: **249** newly derived efficiencies; **1,787** numerical efficiency
  values after retaining source-reported values.
- Positional probability matrices: **14,368 source/profile/position rows**,
  with 20 amino-acid columns. These cover 1,775 original MEROPS memberships,
  18 PICS enzyme/library strata and 3 weighted qPISA experiment/enzyme strata.

## Numerical interpretation

QF percentage activity is relative to the source PLG reference. qPISA depletion
is relative to matched control signal. Neither is silently relabeled as an
absolute molar conversion or kcat. Apparent endpoint/half-life rates assume
exponential loss and record their basis. Fluorescence slopes retain instrument
units; the source selected interval and fit R² are exposed.

PICS frequencies assume approximately mixed input and use observed occurrences.
qPISA candidate frequencies use positive fractional signal depletion as continuous
weights, without a log2 threshold; products and repeated EV6 time points do not
add counts. They remain labeled as candidate evidence. The source query rows
contribute to these descriptive profiles; no claim of independent predictive
validation is made. Partial/nonstandard windows retain coverage metadata.

## Execution scope

- Existing nine final conversion cells were executed with actual data and saved
  numeric output tables.
- Both new merge notebooks were executed; the common-value preview also includes
  source-normalized QF activities and BRENDA efficiencies.
- Earlier notebook analyses and figures were preserved and not rerun in full.
- `ruff check`, `ruff format --check` and Python compilation cover the merge
  package. No pytest suite or full repository build was run.
- Data audits used `/tmp/verify_wide_data.py`, `/tmp/check_wide_edge_sources.py`,
  `/tmp/rerun_merge_notebook.py`; logs are `/tmp/wide_validation.log`,
  `/tmp/wide_edge_sources.log`, `/tmp/merge_notebook_final.log`.
- Notebook execution log: `/tmp/wide_notebooks.log`.

Runtime on the full local backend was approximately 20 seconds to load and 69
additional seconds to construct the wide table during concurrent execution.
The reader currently materializes the dataset in memory. Hash joins and grouping
avoid comparisons between all pairs of source observations; probability scoring
uses dictionaries over eight positions. Time is O((N+L) log(N+L)+C+8R) and memory
O((N+L)Q+C), for N measurements, L links, Q fixed feature columns, C profile cells
and R scored rows. Performance optimality is unknown; projection of unused
provenance columns is a possible future memory improvement.
