# Shared half-life contract — version 4.0

The normalized Parquet, every source protocol CSV and each final condition CSV
have **the same 23 columns**, in this order. One row is one measurement, including
class-only or unresolved observations. Protocol decision flags are transient
notebook diagnostics; they are not exported as extra columns.

| # | Column | Type / meaning |
|---|---|---|
| 1 | `source` | List of all databases contributing this verified measurement. |
| 2 | `source_record_id` | List of `{source, id}` references. `id` is `view:original_id[:measurement_n]`, preserving author subset and split readout identity. |
| 3 | `sequence` | Uppercase sequence with whitespace/HTML removed. No clipping or deletion of nonstandard residues. |
| 4 | `sequence_length` | Integer length of the normalized sequence representation; for nonstandard notation this is not a verified residue count. |
| 5 | `chirality` | Reported chirality; null when unknown. |
| 6 | `topology` | Reported topology, e.g. linear/cyclic; null when unknown. |
| 7 | `n_terminal_modification` | Reported N-terminal modification; `free` is distinct from unknown/null. |
| 8 | `c_terminal_modification` | Reported C-terminal modification. |
| 9 | `other_modifications` | Other documented chemical modifications; `free` means explicitly unmodified. |
| 10 | `species` | Parsed experimental species, e.g. human/mouse; null if unknown. |
| 11 | `matrix` | Experimental matrix, e.g. serum/plasma/other/mixed. |
| 12 | `matrix_fraction` | Explicit serum/plasma volume fraction, 0–1; unknown is null, not 1. |
| 13 | `temperature_c` | Reported incubation temperature in °C; no reference imputation in this field. |
| 14 | `ph` | Reported experimental pH; never sequence pI or an imputed reference. |
| 15 | `peptide_concentration_um` | Explicit single molar concentration in µM. Mass concentrations/doses without validated conversion inputs remain null. |
| 16 | `half_life_seconds` | Positive reported numerical point, or representative midpoint of an explicitly documented CI if there is no reported point. Ordinary ranges, bounds, approximations and qualitative labels remain null. |
| 17 | `sd_seconds` | Explicitly documented SD only; unknown ± error is not SD. |
| 18 | `ci_lower_seconds` | Reported CI lower bound in seconds. |
| 19 | `ci_upper_seconds` | Reported CI upper bound in seconds. |
| 20 | `ci_level` | Documented confidence level, e.g. 0.95. Missing uncertainty is not estimated. |
| 21 | `replicate_count` | Reported experimental replicate count; not the number of database copies. |
| 22 | `half_life_class` | `short`, `medium`, `long`, or null if the evidence cannot support one class. |
| 23 | `in_vivo_in_vitro` | Reported or explicitly verified-linked experimental mode; null if unknown. |

Classes are **short <30 min**, **medium 30 min–6 h inclusive**, **long >6 h**.
For bounds/ranges, assign a class only when the entire supported range lies in
one class. A reviewed project heuristic assigns PEPlife2 2686 (`>1 h`, in vivo)
to medium; the publication does not supply a 6 h upper bound. Its exact value
stays null. See `normalized/remarks_peplife2.md` for the specific rationale.
No class is invented from an unqualified description such as “stable”.

`30–90 min` is class-only medium, not a numerical point or a CI. Unknown ± is
preserved in the per-source remarks, not silently converted. A documented CI
midpoint is a derived representative, not a measured mean, and is specifically
noted. Missing CI remain null; the protocol has a TODO cell for a later estimator.

## Formats and locations

- `normalized/observations.parquet`: typed lists, nullable strings/floats/integers;
  consolidated across all conditions, before any acceptance gates.
- `processed/<condition>/sources/<source>.csv`: the accepted observations that
  include this source in their source list, retaining all known contributors.
- `processed/<condition>/<condition>.csv`: deduplicated union of those source CSVs.

CSV represents the two list columns as JSON and null scalars as blank cells.
Use `proteolysis_data_manager.integration.half_life.schema.read_csv` to decode them.
Parquet uses native `list<string>` and `list<struct<source:string,id:string>>`.
The schema version lives in Parquet metadata and centralized manifests, not in
a repeated per-row column. A source reference example is
`{"source":"dramp","id":"stability:DRAMP01234:measurement_2"}`.

## Shared observations and enrichment

Normalization merges only verified representations: matching source records,
or PlifePred ID + name + full recorded chemistry with an unchanged PEPlife point
label. It unions **all sources and record IDs**, including local repetitions.
Compatible known values fill nulls. Donor IDs and enriched fields are listed in
`normalized/remarks_<source>.md`. Known conflicts stay as separate observations;
there is no averaging, source-priority overwrite or sequence-only merging.

Aggregation removes repeated copies of the same normalized measurement only
when source references overlap and scientific fields match. Coincident points
without verified identity remain separate and appear in notebook diagnostics.
No additional provenance/report columns are added to the final CSV.

## Protocol assumptions and downstream selection

pH and temperature determine acceptance. Missing values use approved references
in transient `_effective` columns; exported reported values stay null. Matrix
fraction, peptide concentration and the transient C/f quantity are available for
later modeling and do not gate acceptance or rescale half-life.

For numerical analysis select `half_life_seconds.notna()`; for class-only
validation select `half_life_seconds.isna() & half_life_class.notna()`. These are
label-availability selections, not precision certification. Source remarks and
raw metadata must still be reviewed for historical label changes, chemistry and
measurement quality. Rows with both label fields null remain available so that
missingness and source coverage are visible.
