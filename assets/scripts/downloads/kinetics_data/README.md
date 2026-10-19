# Kinetics acquisition and normalization

The implementation lives in [proteolysis_data_manager](../../../tools/proteolysis_data_manager/README.md).
This stage uses BRENDA only. No MEROPS matching, peptide localization, cross-family
enrichment or new half-life processing is performed.

## Reuse the acquired BRENDA snapshot

From the repository root, in an environment with the library dependencies:

```bash
python assets/scripts/downloads/kinetics_data/import_brenda.py \
  --archive data/serum_proteolysis_calibration/raw/brenda/brenda_2026.1_json.tar.gz \
  --receipt data/serum_proteolysis_calibration/raw/brenda/download_manifest.json
python assets/scripts/downloads/kinetics_data/normalize_datasets.py
```

The local import preserves exact archive bytes, the original retrieval date and
the source receipt. Reruns verify checksums and keep the first import receipt.
No original legacy file is removed: earlier calibration still uses that snapshot.
Both entry points accept `--data-root`. A new BRENDA release should use a separate
data root rather than overwriting the reviewed release.

The adapter streams all three kinetic families for every EC `3.4.*`, for all
organisms. It does not restrict acquisition to the previous serum protease panel.

## Outputs

```text
data/kinetics_datasets/
  raw/brenda/<unchanged archive>
  normalized/<eight contract tables>.parquet
  normalized/remarks_brenda.md
  processed/human_enzyme/sources/brenda/<eight contract tables>.parquet
  manifests/download_brenda.json
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
[exact join keys and filter semantics](../../../tools/proteolysis_data_manager/docs/KINETICS_CONTRACT.md#protocol-filter-and-meaning-of-the-output).

IDs, not reported names, connect the normalized dictionaries: experiment to
enzyme/substrate through the two relation tables, and measurement to an
explicitly reported cleavage site through its relation table. Avoid flattening
all relations in a single join before aggregation, since that can duplicate
measurements. Final multi-source aggregation is deferred until the first
protocol is reviewed.

See the [contract and relationship diagram](../../../tools/proteolysis_data_manager/docs/KINETICS_CONTRACT.md)
and the [notebook series](../../../experiments/analysis/09_26/20_serum_proteolysis_calibration_max/notebooks/0_data_analysis_kinetics/README.md).
