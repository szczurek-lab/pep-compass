# CLASP analysis (08_26)

Inspection and control figures for the CLASP objective (APEX activity combined
with MEROPS proteolytic-cleavage stability, `-(log2 MIC + lambda * Phi_cleav)`).

## Notebooks (`notebooks/`)

- `clasp_dbaasp_inspection.ipynb` -- inspects the cleavage potential
  `Phi_cleav` against DBAASP peptides, including per-bond and per-protease
  breakdowns and focus enzymes (e.g. aureolysin `M04.009`, V8 / glutamyl
  endopeptidase I `S01.269`).
- `clasp_dbaasp_serum_inspection.ipynb` -- the same inspection restricted to
  the human serum/plasma protease panel (MEROPS dataset id 35).
- `clasp_controls_figures.ipynb` -- control figures relating APEX `log2 MIC`
  activity to cleavage stability across the panel.
- `clasp_cleavage_reduction_scale.ipynb` -- compares the three `CleavagePotential`
  reductions (`sum`, `max_protease`, `max_cut`) on DBAASP peptides, showing how
  the panel-wide `sum` differs in scale (and ranking) from the single-protease /
  single-cut `max_*` modes.

## Dependencies

These notebooks use the migrated CLASP components:

- Proteolysis environment:
  `pep_compass.optimization.components.helpers.proteolysis`
  (`load_protease_panel`, `CleavagePotential`, `MEROPS_AMINO_ACIDS`).
- APEX activity predictor:
  `pep_compass.optimization.components.oracles.strategies.apex_original`
  (`PredictorAPEX`, `onehot_encoding`).

They read MEROPS specificity matrices from `data/merops` at the repository root
and require the APEX `full` model ensemble to be available for the activity
cells.
