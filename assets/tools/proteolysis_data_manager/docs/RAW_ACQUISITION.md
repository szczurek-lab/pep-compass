# Raw acquisition

Rules shared by the three raw-data series: half-life, window neighbourhoods and kinetics.
Acquisition stores source files unchanged. It imports no integration adapters and performs no unit
conversion, label interpretation, record merging or assay filtering.

## Scope

Series-specific commands, sources and storage layouts are described in:

- [Half-life workflow](HALF_LIFE_WORKFLOW.md)
- [Window-neighbourhood workflow](WINDOW_NEIGHBOURHOOD_WORKFLOW.md)
- [Kinetics workflow](KINETICS_WORKFLOW.md)

Launcher paths are listed in the [launcher reference](../scripts/README.md).

## Commands

From the repository root:

```bash
.venv/bin/python assets/scripts/downloads/proteolysis_data_manager/download_all.py --type half_life
.venv/bin/python assets/scripts/downloads/proteolysis_data_manager/window_neighbourhood_data/download_all.py
.venv/bin/python assets/scripts/downloads/proteolysis_data_manager/kinetics_data/download_all.py
```

Omitting `--type` acquires all three series. The installed equivalent is `proteolysis-download-raw`.

### Options

| Option | Effect |
|---|---|
| `--source NAME` (repeatable) | Selects half-life sources in the top-level command |
| `--window-source`, `--kinetics-source` | Select sources of the other two series in the top-level command |
| `--half-life-root`, `--window-root`, `--kinetics-root` | Override the destination of one series |
| `--force` | Reacquires files instead of validating and reusing them |

## Destinations and receipts

Each series has one acquisition receipt, `data/proteolysis/<series>/raw/manifest.json`, with
`<series>` equal to `half_time_datasets`, `window_neighbourhood_datasets` or `kinetics_datasets`.

The `sources` mapping records source versions, file URLs, byte counts, SHA-256 digests and
retrieval times. Local imports add `imported_at` and `imported_from`; unknown original retrieval
dates remain null. Reruns keep the original file receipts. Normalization and protocol manifests are
separate from the acquisition receipt.

### Reruns

A source whose files match the manifest checksums and URLs is validated locally and skipped without
network access. The common raw writer stages a complete source snapshot before replacing files.

## Sources

| Source | Unchanged source input | Acquisition validation |
|---|---|---|
| PEPlife2 | Natural and Modified API JSON plus complete FASTA | API status/count/unique IDs; partition union exactly matches FASTA IDs |
| PepMSND | GitHub `Dataset.xlsx`, pinned revision `c0fa6957c7f88a57af29410cfc075542c92a2c63` | ZIP integrity and original required worksheet columns |
| PlifePred | Published natural/mixed XLSX supplements | ZIP integrity and original required worksheet columns |
| DRAMP | Official `stability_amps.xlsx` | ZIP integrity and original required worksheet columns |
| Cavaco | Versioned PMC article XML/PDF and DOCX/PDF supplements | JATS article structure, PDF signature and DOCX ZIP integrity |
| BRENDA | Official JSON tar archive obtained after licence acceptance | Readable archive, one JSON member, release/version/data header, provenance checksum |
| SABIO-RK | Original paginated Export API JSON for `ECNumber:3.4.*` | Complete pagination, consistent totals, unique entry IDs, no HTML responses |
| MEROPS | Complete existing `data/merops` tree prepared by Karol: index, codes, CSV, NPY, PT and README | Dataset/row identities, axes, matrix dimensions and CSV/NPY count agreement |
| MMP/PICS | Versioned PMC article XML, six XLSX supplements and WebPICS ZIP | JATS article structure and original container integrity |
| DPP4/qPISA | Versioned PMC article XML and eight XLSX supplements | JATS article structure and original container integrity |

### BRENDA licence

The [official BRENDA download](https://www.brenda-enzymes.org/download.php) requires licence
acceptance. The pipeline imports the archive obtained this way and records the operation as a local
import, not as a network download. Import commands are in the [kinetics workflow](KINETICS_WORKFLOW.md#brenda).

### Cavaco

Acquisition retrieves the original articles only. The reviewed 16-value transcription belongs to
integration (see [HALF_LIFE_DATA_DICTIONARY.md](HALF_LIFE_DATA_DICTIONARY.md)).
