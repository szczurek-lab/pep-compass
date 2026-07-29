# E — Insertion mode: post-hoc reweight vs Jacobian whitening

## Question

Is it enough to keep Euclidean MUTANG and bias sampling/selection by
\(d_M^2\), or do we need the brief’s pullback insertion
(\(M\) inside the geometry / whitened Jacobian)?

## Background

Two operationalizations of “similarity metric biases which substitutions the
local search proposes first”:

| Mode | Mechanism | Cost |
| --- | --- | --- |
| **Reweight** | \(\mathrm{SVD}(J)\) unchanged; sample/sort with \(p\propto e^{-d_M^2/\tau}\) | Trivial |
| **Whiten** | \(\tilde J=(I_L\otimes\sqrt{M})J\); SVD defines the pool | One \(\sqrt{M}\) + matmul per Jacobian |

Brief: insert \(M\) into the PepCompass pullback in place of the identity —
that is the whitening / \(\hat J_M\) story. Reweight is a pragmatic ablation.

## Hypotheses

1. For single-position proposals, reweight and whitening produce similar
   short-move enrichment (high rank agreement on a shared candidate universe).
2. Whitening matters more when MUTANG uses several singular directions /
   multi-site coupling (geometry changes which *positions* light up).
3. \(\tau\) in reweight is a new knob that partially mimics \(\lambda\) / \(\alpha\);
   report it explicitly so it is not confused with \(\gamma\).

## Method

1. Define a **universe** per parent: union of Euclidean and whitened pools at
   generous thresholds (or all 19\(L\) singles as an upper bound for short
   peptides).
2. Produce ranked lists:
   - Euclidean singular scores (MUTANG’s native ranking, if available)
   - Euclidean pool reweighted by \(d_M^2\)
   - Whitened MUTANG ranking
3. Compare top-\(k\) sets and rank correlations on the universe.
4. Softmax-sample \(n\) mutants from each policy; report mean \(d_M^2\), mean
   BLOSUM, mean \(|\Delta\log_2\mathrm{MIC}|\) proxy (APEX) — triage only
   (brief §8).

## Metrics

| Metric | What it answers |
| --- | --- |
| Top-\(k\) Jaccard (reweight vs white) | Practical equivalence |
| Kendall \(\tau\) on universe | Global agreement |
| Sampled-batch summaries (\(d_M^2\), BLOSUM, APEX \(\Delta\)) | Downstream flavor |
| Wall time per parent | Engineering cost |

## Success / decision criteria

- Top-\(k\) Jaccard \(> 0.8\) across parents → ship **reweight** for early
  CLASP ablations; keep whitening on the roadmap for full pullback parity.
- Large disagreement on *which positions* are proposed → whitening is not
  optional if we claim metric-aware pullback geometry.
- Either way, document the chosen mode in the ambient API so PoGS / LE-BO
  share one geometry with the combination filter (brief §5).

## Deliverables

- Head-to-head table + recommendation.
- If reweight wins for v1: specify default \(\tau\) and that \(\lambda\) still
  scales \(M\) when used for distances / geodesics elsewhere.

## Dependencies

- Plan D outputs (pools, calibration).
- Optional APEX calls for triage metrics only.
