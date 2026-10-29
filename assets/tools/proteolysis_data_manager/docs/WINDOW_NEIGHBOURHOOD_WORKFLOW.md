# Acquire original cleavage-neighbourhood datasets

This workflow preserves Karol's original MEROPS dictionaries and matrices,
and published MMP/PICS and DPP4/qPISA artifacts. A separate CSV joins the
MEROPS dictionaries and supplied matrix values without changing the source files.

## Run acquisition

From the repository root:

```bash
.venv/bin/python assets/scripts/downloads/proteolysis_data_manager/window_neighbourhood_data/download_all.py
```

The canonical launcher is in this library's `scripts/downloads/window_neighbourhood_data`.
The repository entry resolves through the existing directory symlink. Download
and validation logic lives in `acquisition/window_neighbourhood/<source>.py`;
each module exposes `acquire(directory, ...)`. `download_all.py` selects sources
and assigns their destinations.

Repeat `--source merops`, `--source mmp_pics` or `--source dpp4_qpisa` to select
sources. `--data-root PATH` overrides the default
`data/proteolysis/window_neighbourhood_datasets`. `--force` reacquires files;
ordinary reruns check registered checksums and validate local files without
network requests. For MEROPS, `--force` reimports the original local tree.
`--merops-source PATH` changes its origin from the default `data/merops`.
The installed equivalent is
`proteolysis-download-window-neighbourhood`.

## Original artifacts

| Source | Original location | Retained files |
|---|---|---|
| MEROPS | Existing repository `data/merops` prepared by Karol | The complete tree: `index.json`, `README.md`, 36 dataset folders with `codes.json`, original per-code CSV, counts/logprob NPY and PT files |
| MMP/PICS | [PMC4777984](https://pmc.ncbi.nlm.nih.gov/articles/PMC4777984/), versioned object prefix `PMC4777984.1` | Article XML, `mmc4.xlsx`–`mmc9.xlsx`, `mmc10.zip` |
| DPP4/qPISA | [PMC11612144](https://pmc.ncbi.nlm.nih.gov/articles/PMC11612144/), versioned object prefix `PMC11612144.1` | Article XML, `44320_2024_71_MOESM2_ESM.xlsx`–`44320_2024_71_MOESM9_ESM.xlsx` |

PMC artifacts come directly from
`https://pmc-oa-opendata.s3.amazonaws.com/<version>/<filename>`.
These are original published tables and associated article metadata; acquisition
does not reconstruct tables from processed local matrices or download instrument
data. XLSX sheets and ZIP members remain in their original containers.

MEROPS is a local import of an already prepared dataset. Acquisition retains
all filenames, folder names, dataset IDs, peptidase rows, axis orders and bytes.
It does not regenerate these matrices from the official `Substrate_search`
exports. Validation checks the two index dictionaries, unique contiguous matrix
rows per dataset, matching axis labels/dimensions, finite values and agreement
between the supplied per-code CSV counts and NPY arrays. PT files are copied
as opaque original files; Torch is not required.

The original `data/merops` is read only. A first import requires that directory
or an explicit `--merops-source PATH`; the repository's existing MEROPS installer
can populate it from an explicitly supplied source repository. No fixed public
URL or original download date is invented. Registered raw snapshots can be
reused after the import directory is no longer available.

## Storage and inspection

```text
data/proteolysis/window_neighbourhood_datasets/
  raw/
    manifest.json
    merops/
      index.json
      README.md
      00_a_baumannii_atcc_19606/<unchanged files>
      ...
      35_human_serum/<unchanged files>
    mmp_pics/<eight deposit artifacts plus Activesite.pdf>
    dpp4_qpisa/<nine original artifacts>
  processed/
    merops_matrices.csv
```

There is one `raw/manifest.json` for the whole series. It contains source
versions, requested URLs, byte counts, SHA-256 digests and retrieval times.
Local imports record their import time separately and retain known original
retrieval metadata. Complete source candidates are validated before replacing
registered files. XML and XLSX containers are checked; the WebPICS ZIP is
checked for integrity. No unit conversion, assay filtering, deduplication or
cross-source matching is performed.

The MMP/PICS source also acquires `Activesite.pdf`, the primary-paper copy from
the University of Oviedo repository (DOI `10.1016/j.matbio.2015.09.003`). It
defines the synthetic QF substrate chemistry and assay conditions. The original
`mmc8.xlsx` already contains QF time courses, per-round `k` coefficients and
activity percentages relative to PLG↓L; these values are not converted into
kcat, Km or kcat/Km during acquisition.

## Merge MEROPS into CSV

```bash
.venv/bin/python assets/scripts/integration/proteolysis_data_manager/window_neighbourhood_data/merge_merops.py
```

The launcher invokes `integration.window_neighbourhood.merops.merge_merops`.
It verifies acquired file checksums and writes the complete CSV separately from
`raw`. `--data-root PATH` selects the series; `--output-path PATH` selects the
CSV destination. The installed equivalent is `proteolysis-merge-merops`.

One row represents an original matrix cell. The unique key is
`(dataset_index, peptidase_row, position_index, amino_acid_index)`.
Columns are:

```text
dataset_index, folder, scientific, role, apex_index, label, taxids,
peptidase_row, code, name, n_cleavages, serum_category,
position_index, position, amino_acid_index, amino_acid, count, logprob
```

`taxids` is a JSON list inside a CSV cell; absent `apex_index` and
`serum_category` values are empty. Position labels retain the source spelling
(`P1p`, etc.). Dataset IDs and peptidase/axis indices refer to the unchanged
JSON/NPY dictionaries. Both numeric fields copy existing arrays; log-probabilities
are not recomputed or rounded for presentation. Repeated codes and strain-level
dataset memberships remain separate. The existing snapshot contains 1775
dataset/peptidase matrices, producing 284000 cells (1775 × 8 × 20), not 284000
independent cleavage observations. No cross-source biological merge is performed.

The earlier `Substrate_search.txt`/`.sql` exports and their original source
receipt are retained under
`tmp/archive_old_analysis/window_neighbourhood_merops_substrate_search/`.
They are no longer the active MEROPS dataset.

Open the [raw inspection notebook](../../../experiments/pcpsackm/datasets_analysis/window_neighbourhood/acquisition_analysis.ipynb).
It runs acquisition, displays original MEROPS index entries, codes and a selected
matrix, runs/loads the separate MEROPS CSV, and displays original XLSX sheets.
It also reads article XML as text and lists the WebPICS ZIP members.
Pandas displays are previews; the files on disk retain all source content.
