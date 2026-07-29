# D — Arm M1: MUTANG under enhanced pullback

## Goal

Measure how enhanced ambient \(M\) changes **MUTANG proposals**, for MDS
metrics from BLOSUM62 and CLASP/AMP-BLOSUM (plan A), vs Euclidean \(M=I\).

LAMS is **off** in this arm (or: materialize the bounded product without the
LAMS cosine gate) so proposal effects are not confounded by combination filtering.

## Method

1. Parent panel (≥30 AMPs; include `GLLKRIKTLL`).  
2. For each parent, decoder Jacobian \(J\) (HydrAMP, same settings as POC).  
3. Conditions:
   - `euclid`: \(\mathrm{SVD}(J)\) → MUTANG map  
   - `white[M]`: \(\mathrm{SVD}(\tilde J)\), \(\tilde J=(I_L\otimes\sqrt{M})J\), for
     \(M_{62}\) and \(M_{\mathrm{CLASP}}\) from A  

4. Match median single-position proposal count across conditions (retune
   `token_threshold` or cap top-\(k\) directions).  
5. Optional secondary: Euclidean pool **re-ranked** by \(d_M^2\) (no whitening) —
   only if needed to separate “new proposals” from “reordering” (see
   `E_insertion_mode.md`).

## Metrics

| Metric | Purpose |
| --- | --- |
| Jaccard(euclid, white) of single-AA proposals | Set shift |
| Spearman ranks on intersection | Reorder strength |
| Mean \(d_M^2\), mean BLOSUM score of proposals | Conservative enrichment |
| Pool size vs `token_threshold` | Calibration |
| Position entropy of proposals | Collapse onto few sites? |

Compare \(M_{62}\) vs \(M_{\mathrm{CLASP}}\) vs Euclidean.

## Success / readout

- Does whitening enrich conservative swaps at matched pool size?  
- Does CLASP \(M\) differ from BLOSUM62 on proposal composition?

## Deliverables

- Summary table over parents × \(M\)  
- Case-study parents with largest euclid↔white disagreement  
- Recommended default \(M\) for MUTANG-only use  

## Dependencies

- Plan A shortlist  
- POC `whiten_jacobian` / HydrAMP + `MutationEnumerationInTangentSpace`
