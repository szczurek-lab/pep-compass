# Kinetics acquisition and normalization

The implementation lives in [proteolysis_data_manager](../README.md).
Raw acquisition supports BRENDA and SABIO-RK. The existing normalization and
protocol below apply to BRENDA only. SABIO-RK remains in its original format.

## Acquire the raw sources

From the repository root:

```bash
.venv/bin/python assets/scripts/downloads/proteolysis_data_manager/kinetics_data/download_all.py
```

The default root is `data/proteolysis/kinetics_datasets`. The command validates
and reuses the registered BRENDA archive, then retrieves all SABIO-RK pages.
Use `--source sabio` or `--source brenda` to acquire one source, `--data-root PATH`
to change the destination, and `--force` to reacquire registered sources.
The canonical scripts are in the library's `scripts/downloads/kinetics_data`;
the repository path above resolves through a directory symlink.

### SABIO-RK

The [official Export API](https://sabiork.h-its.org/ui/export-api) request is
`GET https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/json` with
`q=ECNumber:3.4.*`, `page=1` and `pageSize=1000`. Acquisition reads the first
page's `meta` to discover the complete page range. It retains each response
unchanged as `raw/sabio/page_000001.json`, etc., including its `meta` and `data`.
The default query includes all peptidase EC numbers and organisms. No assay,
organism, numerical-value or unit filter is applied after retrieval.

`--sabio-query EXPRESSION` changes the explicit source query;
`--sabio-page-size N` accepts 1–1000. Use a separate root for a different
query snapshot, or `--force` to replace it. The requested URLs in the shared
manifest retain the complete query. The adapter has not yet been defined:
parameter types, units, bounds, conditions and associations remain as published.

Before publication, acquisition rejects HTML/error responses, missing or
truncated pages, inconsistent pagination totals and duplicate entry IDs.
Ordinary reruns validate the registered pages locally without network requests.
The old `/sabioRestWebServices/` endpoints are not used; they redirected to the
HTML interface during implementation.

### BRENDA

The [official download](https://www.brenda-enzymes.org/download.php) requires
licence acceptance. For a first import, supply the already obtained archive:

```bash
.venv/bin/python assets/scripts/downloads/proteolysis_data_manager/kinetics_data/download_all.py \
  --brenda-archive PATH_TO_LICENSED_ARCHIVE \
  --brenda-receipt PATH_TO_ORIGINAL_RECEIPT \
  --brenda-release 2026.1
```

The receipt is optional when no original provenance exists; acquisition then
leaves the original retrieval time unknown. The full official BRENDA archive is
retained, including EC classes outside `3.4.*`. Peptidase selection belongs to
the existing normalization stage. No new BRENDA network retrieval is claimed.

## Reuse the acquired BRENDA snapshot

To import BRENDA independently and run its existing normalization:

```bash
python assets/scripts/downloads/proteolysis_data_manager/kinetics_data/import_brenda.py \
  --archive PATH_TO_LICENSED_ARCHIVE \
  --receipt PATH_TO_ORIGINAL_RECEIPT
python assets/scripts/integration/proteolysis_data_manager/kinetics_data/normalize_datasets.py
```

The local import preserves exact archive bytes and the original retrieval date.
It stores one acquisition receipt at `raw/manifest.json`, including the original
checksum, source URL and known retrieval time; the original receipt is not copied
into another metadata file. Reruns verify checksums and keep the first import receipt.
No original legacy file is removed: earlier calibration still uses that snapshot.
Both entry points accept `--data-root`. A new BRENDA release should use a separate
data root rather than overwriting the reviewed release.

The adapter streams all three kinetic families for every EC `3.4.*`, for all
organisms. It does not restrict acquisition to the previous serum protease panel.

## Outputs

```text
data/proteolysis/kinetics_datasets/
  raw/brenda/<unchanged archive>
  raw/sabio/page_000001.json
  raw/sabio/<remaining original response pages>
  raw/manifest.json
  normalized/<eight contract tables>.parquet
  normalized/remarks_brenda.md
  processed/human_enzyme/sources/brenda/<eight contract tables>.parquet
  manifests/normalization_brenda.json
  manifests/protocol_human_enzyme_brenda.json
```

The first review protocol selects contexts whose linked enzyme species are all
documented as `Homo sapiens`. Its `human_enzyme` output does not mean human
serum/plasma: assay sample species, matrix, and in-vitro mode are not currently
filters, while pH and temperature bounds are disabled. It retains every
measurement in selected contexts, including nonnumeric measurements and
measurements with undocumented conditions. This output is a distribution of
BRENDA records associated with human-enzyme contexts, not a serum-specific
distribution or a count of independent experiments. See the contract for the
[exact join keys and filter semantics](KINETICS_CONTRACT.md#protocol-filter-and-meaning-of-the-output).

IDs, not reported names, connect the normalized dictionaries: experiment to
enzyme/substrate through the two relation tables, and measurement to an
explicitly reported cleavage site through its relation table. Avoid flattening
all relations in a single join before aggregation, since that can duplicate
measurements. Final multi-source aggregation is deferred until the first
protocol is reviewed.

The [raw inspection notebook](../../../experiments/pcpsackm/datasets_analysis/kinetics_data/acquisition_analysis.ipynb)
runs acquisition and displays original BRENDA and SABIO-RK records. Its selected
BRENDA EC is only a bounded preview of the full archive. It performs no normalization.
See the [contract and relationship diagram](KINETICS_CONTRACT.md) for the
existing BRENDA integration.
