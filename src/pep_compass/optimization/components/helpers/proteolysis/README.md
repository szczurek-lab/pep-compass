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

## Cleavage aggregation (`reduction`)

`CleavagePotential` (and `build_clasp_potential`) accept a `reduction` argument
that controls how the per-bond, per-protease scores collapse into a single
susceptibility `Phi_cleav`. It is orthogonal to `variant`
(`product` / `additive` / `meanfield`), which only sets the scoring math:

- `"sum"` (default) -- weighted sum over the whole panel,
  `Phi = sum_pi rho_pi(.)`, with bonds combined as `variant` dictates. This is
  the original panel-wide behaviour: susceptibility grows with the number of
  proteases and the number of cuttable bonds.
- `"max_protease"` -- keep only the single dominant protease,
  `Phi = max_pi rho_pi(.)`. Susceptibility is set by the *most aggressive
  enzyme* rather than the summed panel; bonds are still combined per `variant`.
- `"max_cut"` -- the single most-likely cut event: maximum over both proteases
  and bonds, `Phi = max_pi max_b rho_pi R(pi, b)`. This overrides the variant's
  over-bond aggregation and isolates the weakest bond against the strongest
  protease.

Because `"sum"` mixes contributions across all proteases and bonds, its
magnitude scales with panel size and peptide length, whereas the two `max_*`
modes stay on the scale of a single protease/bond. See the analysis notebook
`assets/experiments/analysis/08_26/20_clasp_analysis_karol/notebooks/`
`clasp_cleavage_reduction_scale.ipynb` for a side-by-side comparison of the
three scales.

## Data

`load_protease_panel` reads from `data/merops` at the repository root by
default (override with `merops_root`). The MEROPS datasets are expected to
contain `index.json` plus per-dataset `matrices_logprob.npy` and `codes.json`.

`data/*` is gitignored, so fetch the curated matrices with
`assets/scripts/downloads/merops/download_merops_matrices.sh` (mirrors the
APEX / HydrAMP model download scripts; provide the source via
`MEROPS_REPOSITORY`).
