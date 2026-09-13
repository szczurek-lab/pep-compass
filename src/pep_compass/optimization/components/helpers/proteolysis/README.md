# Proteolysis environment

Reusable environment for modelling proteolytic (exo- and endopeptidase)
cleavage of peptides from MEROPS protease specificity matrices.

## What lives here (and why)

This package holds the **environment / physics**, not a filter:

- `ProteasePanel`, `load_protease_panel` -- load and concatenate MEROPS
  specificity matrices (`matrices_logprob.npy`) with per-protease weights and
  codes for one or more datasets.
- `CleavagePotential` -- the proteolytic-susceptibility potential
  `Phi_cleav` over the eight-residue window `P4 P3 P2 P1 | P1' P2' P3' P4'`
  around each bond. It scores concrete peptides (`sequence_log_potential`), a
  decoder's differentiable residue distribution (`distribution_log_potential`),
  and exposes per-bond / per-protease breakdowns for analysis.

It is placed under `components/helpers` because the same physics is consumed by
several components:

- the ranked CLASP scoring objective
  (`components/filters/ranked/scoring/clasp.py`),
- oracles / black boxes that need a differentiable stability term,
- analysis notebooks.

The complementary **ranking objective** that combines this cleavage-stability
term with APEX activity (`ClaspPotential`, `build_clasp_potential`,
`compose_clasp_distribution`) is a scoring concern and therefore lives in
`components/filters/ranked/scoring/clasp.py`, not here.

## Data

`load_protease_panel` reads from `data/merops` at the repository root by
default (override with `merops_root`). The MEROPS datasets are expected to
contain `index.json` plus per-dataset `matrices_logprob.npy` and `codes.json`.
