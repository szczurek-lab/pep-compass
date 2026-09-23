# Serum proteolysis calibration acquisition

These scripts acquire immutable source artifacts for serum-proteolysis
calibration. They implement the source restrictions in
`tmp/clasp/serum_data_acquisition_section.md`. They do not parse kinetic
records, infer peptide sequences, convert units, or construct model-ready
tables.

The source registry is kept with the acquisition code at
`config/serum_sources.yaml`. This directory is self-contained; no repository-
level configuration file is required.

## Scope

The local MEROPS panel is the only protease universe. It has 55 entries;
`C111.001` (factor XIIIa) is retained in the manifest but excluded from the
peptide-bond-hydrolysis model. All kinetic API requests therefore target only
mapped entries among the remaining 54 proteases.

External sources have distinct roles:

- BRENDA and SABIO-RK provide curated kinetic records.
- PXD002265 provides MMP cleavage specificity observations.
- PXD042089 provides processed DPP4 qPISA data.
- qMSP-MS and ADAMTS13 provide selected published quantitative tables.
- PEPlife2 and DRAMP provide peptide stability observations.
- Human Protein Atlas provides abundance priors, not active protease
  concentrations.

## Storage layout

All source artifacts remain below one local root:

```text
data/serum_proteolysis_calibration/
├── manifests/serum_proteases.tsv
├── reports/unresolved_serum_proteases.tsv
└── raw/
    ├── brenda/
    ├── uniprot/
    ├── sabio/<MEROPS_CODE>/
    ├── pride/<PXD>/
    ├── mmp_pics/
    ├── dpp4_qpisa/
    ├── qmsp_ms/
    ├── adamts13/
    ├── peplife2/
    ├── dramp/
    └── hpa/
```

Each downloader retains a raw response unchanged and writes a
`download_manifest.json` receipt with source URL or query, retrieval time,
file size, and SHA-256 checksum.

## Required order

Build the fixed manifest first:

```bash
uv run assets/scripts/downloads/serum_proteolysis_calibration/build_serum_manifest.py
```

It creates `data/serum_proteolysis_calibration/manifests/serum_proteases.tsv`.
The manifest initially has blank UniProt and EC fields. Do not edit its
MEROPS-derived fields manually.

Install the BRENDA archive manually after accepting its licence in a browser:

```bash
uv run assets/scripts/downloads/serum_proteolysis_calibration/install_brenda_bulk_export.py \
  data/serum_proteolysis_calibration/brenda_2026_1.json.tar.gz \
  --format json \
  --release 2026.1
```

Run the complete restricted acquisition:

```bash
uv run assets/scripts/downloads/serum_proteolysis_calibration/download_serum_sources.py
```

The orchestrator reads the colocated registry, executes each configured
source in dependency order, and writes the command and return code for every
attempt to `data/serum_proteolysis_calibration/raw/acquisition_run_manifest.json`.
Use `--config` only when deliberately testing an alternative registry.

The command maps MEROPS codes through UniProt before calling SABIO-RK. It
reports unresolved mappings without guessing an enzyme identity. A required
source failure produces a non-zero exit status and is recorded in
`raw/acquisition_run_manifest.json`.

Run a single source when diagnosing a failure:

```bash
uv run assets/scripts/downloads/serum_proteolysis_calibration/download_serum_sources.py \
  --source sabio
```

## Source-specific restrictions

SABIO-RK requests are constructed as:

```text
Organism:"Homo sapiens" AND ECNumber:<EC_NUMBER>
```

The downloader first obtains kinetic-law IDs, then requests normalized SBML
only for those IDs. A response that is not SBML is rejected. An empty ID list
is preserved as an explicit empty source result; it is not replaced with a
broader name search.

PRIDE commands use dataset profiles. They download processed files only:

```bash
uv run assets/scripts/downloads/serum_proteolysis_calibration/download_pride_project.py \
  PXD002265 --profile mmp_pics

uv run assets/scripts/downloads/serum_proteolysis_calibration/download_pride_project.py \
  PXD042089 --profile dpp4_qpisa
```

Vendor `.raw`, `.mzML`, `.mzXML`, and `.wiff` files are rejected unless an
explicit raw-MS workflow is implemented. `--all-files` is intentionally not
available.

PMC source acquisition uses the public PMC Open Data bucket and lists only one
article version before selecting numerical supplements. It does not scrape
publisher HTML or PDFs. The profiles are `mmp_pics`, `dpp4_qpisa`, `qmsp_ms`,
and `adamts13`.

The older blood-serine microarray studies are optional. No downloader attempts
to reconstruct their numeric matrix from figures or PDFs. A manually supplied
machine-readable table belongs in `data/manual/serine_microarrays/` with a
provenance sidecar.

## Next phase

Raw artifacts are not model-ready. The next phase parses the source formats,
maps records to resolved proteases and substrates, preserves experimental
conditions, and writes normalized tables under
`data/serum_proteolysis_calibration/processed/`.
