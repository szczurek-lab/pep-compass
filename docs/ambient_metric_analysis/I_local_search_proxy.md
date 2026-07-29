# I — Local-search proxy runs (in silico, pre–wet-lab)

## Question

In short CLASP-like local trajectories, does adding the ambient metric change
*how* stability is achieved (conservative edits, fewer drastic swaps) compared
with Euclidean geometry at the same \(\gamma\)?

## Background

Brief §6 ablation order:

1. Potential only, Euclidean metric  
2. Add combination filter  
3. Add AMP-BLOSUM metric (BLOSUM62 intermediate)  
4. Vary \(\gamma\) (and \(\lambda\)); Pareto against \(\gamma=0\)  
5. Confirm MUTANG-preferred substitutions coincide with low-\(\Delta\mathrm{cleav}\)

This plan is a **computational** rehearsal of steps 1–4, not wet-lab C2/C3.

## Hypotheses

1. At fixed \(\gamma>0\), metric-aware search reaches similar
   \(\Phi_{\mathrm{cleav}}\) reduction with **higher** mean substitution BLOSUM
   score and lower mean Grantham distance than Euclidean.
2. \(\lambda\) changes the path shape / edit composition more than the final
   \(\Phi\) value (consistent with “does not move the optimum” if the search
   is rich enough; under tight MUTANG budgets it may still affect attained
   \(\Phi\)).
3. \(\gamma=0\) controls show little stability gain regardless of metric
   (metric alone is not a stability objective).

## Method

1. Freeze \((M,\text{insertion mode},\text{filter on/off})\) from A–G.
2. Arms (minimum):
   - \(\gamma=0\), \(M=I\)
   - \(\gamma>0\), \(M=I\)
   - \(\gamma>0\), \(M=M_{62}\)
   - \(\gamma>0\), \(M=M_{\mathrm{AMP}}\)
3. For each of \(N\) parents, run a fixed budget local enumeration /
   short LE-BO-style loop (same seed policy, same candidate cap).
4. Record trajectory summaries and final candidates.

## Metrics

| Metric | Role |
| --- | --- |
| \(\Delta\Phi_{\mathrm{cleav}}\) (final − parent) | Stability objective |
| \(\Delta\log_2\mathrm{MIC}\) proxy (APEX) | Triage only (§8) |
| Mean BLOSUM / Grantham of accepted edits | Conservatism |
| Edit distance / number of sites changed | Path length |
| Fraction of accepted edits with \(\Delta\mathrm{cleav}<0\) | Alignment with §6 step 5 |
| Pareto sketch: activity proxy vs \(\Phi_{\mathrm{cleav}}\) across \(\gamma\) | Brief step 4 |

## Success / decision criteria

- Metric arms show more conservative edit sets at matched \(\Phi\) improvement
  → good computational support for including \(M\) in CLASP.
- If metric arms only help by shrinking the candidate set and miss better
  protective drastic swaps that Euclidean finds → keep metric weak (\(\lambda\)
  small) or behind the \(\Delta\mathrm{cleav}<0\) gate.
- Do **not** treat APEX gains as confirmation of C3.

## Deliverables

- Comparison table across arms; a few trajectory plots.
- Go / no-go for implementing ambient \(M\) in production pullback code.
- Input to wet-lab design: which computational arm’s peptides to prioritize
  if SPOT capacity is limited.

## Dependencies

- Cleavage potential + MUTANG/filter integration.
- Plans A–H defaults frozen.
- Optional: PoGS energy hook with \(\gamma\Phi_{\mathrm{cleav}}\).

## Notes

- Matched pairs for future wet-lab C2 should be drawn from the same parent with
  \(\gamma>0\) ± metric arms held fixed otherwise.
- Keep computational cost linear in \(|\Pi|\) in mind; panel size is a data
  constraint, not a metric constraint (brief §3).
