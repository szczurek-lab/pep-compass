# Half-life source mappings and data flow

The common contract is [CLEAN_SCHEMA.md](../../../experiments/analysis/09_26/20_serum_proteolysis_calibration_max/notebooks/0_data_analysis_half_life/protocols/CLEAN_SCHEMA.md).
It contains 23 fields from normalized Parquet through final CSV.

| Stage | Files / responsibility |
|---|---|
| Acquisition | `raw/<source>/`: unchanged JSON/FASTA/XLSX/publication artifacts. `manifests/download_<source>.json` verifies origin and checksum. |
| Source adapters | `assets/tools/proteolysis_data_manager/src/proteolysis_data_manager/integration/half_life/adapters/`: parse each source and retain source wording in a transient detailed inventory. |
| Normalization | `normalized/observations.parquet`: compact typed observations across all conditions; verified repetitions consolidated and compatible missing fields enriched. No condition filtering. |
| Decision notes | `normalized/remarks_<source>.md`: concrete IDs, original non-exact/error wording, class decisions, publication links, historical conflicts and enrichment donor IDs. |
| Source protocol | `processed/<condition>/sources/<source>.csv`: same compact contract, filtered by species/matrix/mode and pH/temperature. |
| Aggregation | `processed/<condition>/<condition>.csv`: same compact contract; one condition's verified union. |
| Receipts | `manifests/`: all download, normalization, protocol and aggregation manifests. |

`views/` is retained temporarily for preceding calibration work; the revised
normalizer does not rewrite it. No new reports or wide normalized copies are saved.
Source overviews reconstruct the detailed source inventory in memory to inspect
original descriptions and calculate common reference-backbone descriptors.
They do not treat that transient inventory as the public normalized contract.

## Source adapters

| Adapter / raw input | Mapping and verification |
|---|---|
| `peplife2.py`: natural/modified JSON | `id`, `seq`, `half_life` + `units_half`; species/matrix from `test_sample`, mode from `vivo_vitro`; chemistry from `chiral`, `lin_cyc`, `nter`, `cter`, `chem_mod`; concentration from `conc`; explicit incubation pH/temperature from experimental context. Complete FASTA checks the API ID union. |
| `plifepred.py`: natural/mixed XLSX | `id.pdb`, `seq`, `hl in seconds` / `Half-life in seconds`, recorded chemistry. Recover experimental context only when PEPlife ID, name and full recorded chemistry agree. Author labels stay unchanged; discrepancies are listed in remarks. Natural/mixed views remain in source references. |
| `pepmsnd.py`: dataset XLSX | `ID`, `Sequence`, `T1/2(hour)`, `Test Sample`, `In-Vivo/Vitro`, `Cyclic/Linear`, terminal/chemical modifications. No dedicated publication/chirality field: do not reconstruct these from sequence equality. |
| `dramp.py`: stability_amps XLSX | `DRAMP_ID`, `Sequence`, `half_life`, `Type`, optional publication/assay/chemistry. Split explicit half-life/matrix clauses, with a separate measurement suffix and fraction. Recover assay mode only through a unique consistent publication + chemistry + species/matrix link. |
| `cavaco.py`: article + reviewed transcription | 16 printed Figure 3a validation points and reported 95% CI; Table 1 sequences and PDF digest verified. Human serum, in vitro, 37 °C, fraction 0.5, final concentration 500 µM, n=3. Training records not reconstructed. |

Reviewed corrections live in `adapters/curation.py` and are noted per record.
PEPlife2 5066–5069 are mouse **in-vivo PK** measurements, as shown in primary
publication Table 3. The source's overnight 4 °C and 10 µg/mL values describe
ELISA plate coating; they do not become incubation/exposure covariates. Raw
records stay unchanged, normalized mode is corrected and temperature stays null.

## Numeric rules

Time is stored in seconds; explicit molar concentrations in µM; temperature in
°C; dilution as a volume fraction. Administered doses and mass concentrations
are not converted using an unmodified-backbone mass for chemically modified
peptides. The reported sample concentration is not automatically changed by a
mixing ratio unless the source documents its final meaning (Cavaco does).

Reported points remain numerical. Ordinary intervals, bounds, approximations,
remaining-fraction-at-time and activity descriptions do not become exact labels.
`half_life_class` uses short <30 min, medium 30 min–6 h inclusive, long >6 h.
Unresolvable descriptions stay null. The single reviewed `>1 h` project heuristic
is identified explicitly in PEPlife2 remarks and is not presented as an author class.

Missing SD and CI stay null. An unspecified ± is preserved with its raw value in
remarks, not renamed SD. Only explicitly documented CI may supply a representative
midpoint when no point is reported; ordinary range midpoints are forbidden.

## Library entry points

```python
from proteolysis_data_manager.protocols.half_life.filtering import load_half_life_data
from proteolysis_data_manager.integration.half_life.schema import read_csv

observations = load_half_life_data(data_root)  # typed compact Parquet
peplife = load_half_life_data(data_root, "peplife2")  # membership in source list
human_serum = read_csv(data_root / "processed/human_serum/human_serum.csv")
```

All library imports use `proteolysis_data_manager`. Acquisition entry
scripts delegate to that library; there is one implementation of each
adapter/downloader.
