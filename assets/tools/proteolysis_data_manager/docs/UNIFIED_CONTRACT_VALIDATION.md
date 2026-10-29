# Original normalized-backend validation

This is the historical validation of the normalized backend before the wide-feature
revision. Current common-parameter transforms and notebook validation are documented
in `WIDE_PARAMETERS_VALIDATION.md`. The original implementation hashes refer to the
original build snapshot, not to the subsequently reorganized reader modules.

Executed on 2026-10-22 against local branch `dev_max_integrated`, commit `da7b5f0`,
with existing workspace changes. No remote branch refresh was performed under the
restricted network configuration. This report covers the new merge package and
notebook conversion sections; earlier notebook analyses were not rerun in full.

## Executed artifacts

- One database: `data/proteolysis/merged_datasets/proteolysis.sqlite`.
- 13 registered parsers, covering 10 source identities.
- 233,021 measurement records; these include raw signals, reported summaries and
  derived responses and are **not** counts of independent experiments.
- 153,838 context records, not globally unique sequence strings or independent cuts.
- 1,775 original MEROPS profile memberships across overlapping populations.
- 1,462 condition descriptions, 35,931 source-scoped assays, 242,408 participants.

| Source | Measurement records |
| --- | ---: |
| BRENDA | 26,697 |
| SABIO-RK | 5,204 |
| PEPlife2 | 4,500 |
| PepMSND | 635 |
| DRAMP | 111 |
| PlifePred | 424 |
| Cavaco | 16 |
| MMP/PICS, including QF | 30,018 |
| qPISA EV1, EV4 and EV6 | 165,416 |

Original MEROPS marginal cells occupy `profile_cells` and are not counted as
empirical measurements. All 3,299 SABIO source reaction records remain present,
including records without the three selected kinetic parameter types.

## Acceptance criteria

| Criterion | Verdict | Executed evidence |
| --- | --- | --- |
| Separate parsers, common interface | PASS | Inspected signatures of all 13 `parse(root, *, data=None)` functions |
| Shared quantities, units and source identity | PASS | Full reader validation; original BRENDA points independently checked against SI conversions |
| Conditions separate from participant configurations | PASS | Condition/assay key checks; EV1 shares one condition description across two enzyme assays |
| All records retained without join multiplication | PASS | Source counts, independent counts from original BRENDA/SABIO raw records, unique measurement IDs, common values view cardinality |
| Partial and unsupported windows retained | PASS | All contexts retained; coverage/status checks on every feature |
| Requested MEROPS score | PASS | Independently reconstructed every numeric sum, log sum and product from original profile cells; maximum absolute discrepancy 0 |
| qPISA numerical inputs retained | PASS | Reconstructed all 165,176 endpoint responses and 72 time-course changes from stored source inputs |
| Participant doses normalized with originals retained | PASS | Checked every recorded dose/range against `reported_amount_json`; M and kg remain distinct dimensions |
| One SQLite reader and valid serialization | PASS | Full public read, schema/version/quantity checks, `PRAGMA integrity_check`, `PRAGMA foreign_key_check` |
| Existing notebooks connected to parsers | PASS | Executed all nine appended conversion cells; validated notebook structure and saved outputs |
| No notebook integration exports | PASS | Conversion cells contain no CSV or database writes |
| Raw inputs unchanged | PASS | All acquisition source checksums match original manifests; build manifest hashes match |
| Canonical script shortcut | PASS | Global script resolves to the library script through the existing directory symlink |
| Current implementation recorded | PASS | All merge package Python hashes match the SQLite build metadata |

### Score availability

For the explicitly selected human MEROPS population 34:

| Status | Context records |
| --- | ---: |
| `complete_window` | 2,980 |
| `partial_window` | 46,692 |
| `no_explicit_profile_mapping` | 103,950 |
| `unsupported_residue` | 216 |

The last two statuses retain null scores and preserve their empirical measurements.
The unsupported cases are the defined QF substrates containing Dpa. No Dpa-to-standard
residue substitution or inferred EC-to-MEROPS mapping was applied.

## Commands and checks

The final build was executed through the public shortcut:

```bash
.venv/bin/python assets/scripts/integration/proteolysis_data_manager/merged_data/build_dataset.py --score-windows --replace
```

`--replace` was used only to regenerate the derived database created during this
implementation. Raw directories were read and checksum-verified.

Additional executed checks:

- `ruff check` on the new merge package and canonical build script: passed.
- `ruff format --check` on the same 25 Python files: passed.
- `python -m compileall` on those paths: passed.
- The data audit and conversion-cell execution used local temporary scripts:
  `/tmp/verify_unified_contract.py` and `/tmp/execute_contract_notebooks.py`.
  Logs: `/tmp/unified_contract_validation.txt`, `/tmp/contract_notebooks_final.log`.

No pytest suite, full repository build, static type checker or complete earlier
notebook analysis was run. Validation was scoped to the requested data conversion,
reader, numerical invariants and appended notebook sections. Scientific extraction
rules still require source interpretation; invariant checks do not establish that
every source-associated kinetic parameter is a site-specific rate.
