# F — Interaction with \(\Phi_{\mathrm{cleav}}\) (metric selects *among* protective swaps)

## Question

At cleavage-relevant positions, among substitutions that **reduce**
\(\Phi_{\mathrm{cleav}}\), does the ambient metric prefer conservative
replacements — the design rationale in brief §4?

## Background

Brief rationale (paraphrased):

- \(\Phi_{\mathrm{cleav}}\) forces edits at susceptible sites but does not choose
  among protective amino acids.
- Euclidean treats Ile→Leu and Ile→Pro as equal length.
- Similarity metric biases toward Ile→Leu-like moves, independent of APEX.

This is the core *usefulness* test of the metric **as a prior**, before wet-lab
C2.

## Hypotheses

1. For positions with high wild-type bond contribution, the Pareto set of
   (low \(\Delta\Phi_{\mathrm{cleav}}\), low \(d_M^2\)) is nonempty and enriched
   for physicochemically conservative swaps under BLOSUM/AMP metrics but not
   under \(M=I\).
2. AMP-BLOSUM improves that enrichment over BLOSUM62 on AMP parents (link to B).
3. Metric distances are sign-blind: \(d_M(a,b)\) does not predict
   \(\mathrm{sign}(\Delta\Phi)\) — confirming separation of roles (§2).

## Method

1. Implement or reuse a differentiable / exact \(\Phi_{\mathrm{cleav}}\) on one-hot
   or decoder simplex peptides (brief §3: product-of-rates form preferred).
2. Parent panel with measurable predicted susceptibility (filter out already
   “flat” peptides).
3. For each parent×position:
   - Evaluate all 19 substitutions: \(\Delta\Phi\), \(d_M^2\) under \(I\), \(M_{62}\),
     \(M_{\mathrm{AMP}}\) (and pep-ablated defaults from C).
4. Restrict to protective set \(\Delta\Phi < 0\); rank by \(d_M^2\); compare
   composition of top-\(k\) protective shorts vs top-\(k\) protective by
   \(|\Delta\Phi|\) alone.
5. **Negative control:** show Spearman(\(d_M^2\), \(\mathrm{sign}(\Delta\Phi)\)) ≈ 0.

## Metrics

| Metric | What it answers |
| --- | --- |
| Mean BLOSUM score of top-\(k\) protective-by-\(d_M^2\) | Conservative bias among protective |
| Same under Euclidean random / \(|\Delta\Phi|\) ranking | Baselines |
| Fraction of top protective shorts that are “drastic” (hand list or Grantham threshold) | Failure rate |
| Overlap between metric-short protective and max-protective | Does prior fight the potential? |

## Success / decision criteria

- Metric-short protective set has **higher** mean BLOSUM (more conservative)
  than \(|\Delta\Phi|\)-only top-\(k\), without large loss in mean protective
  \(\Delta\Phi\) → supports brief rationale.
- If metric-short protective swaps are much weaker on \(\Delta\Phi\) → \(\lambda\)
  too strong relative to how candidates are filtered; soften prior or apply
  metric only after a \(\Delta\Phi<0\) gate (cf. combination filter §5).
- Confirm sign-blindness; if violated, revisit construction (plan C).

## Deliverables

- Per-metric summary on the parent panel.
- A few annotated position case studies (e.g. aureolysin-like P1′ hydrophobics
  from the brief).
- Recommendation: use metric as (i) soft prior only, or (ii) always behind a
  \(\Delta\Phi<0\) hard gate.

## Dependencies

- Cleavage potential code + protease panel weights \(\rho_\pi\).
- Frozen \(M\) from A–C; proposal machinery optional (can brute-force singles).

## Notes

- APEX is **not** the activity claim here; optional APEX deltas are triage only
  (brief §8). True activity is wet-lab C3.
- Mean-field vs product-of-rates: pick one and keep fixed across metrics.
