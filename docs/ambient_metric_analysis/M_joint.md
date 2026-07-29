# M — Arm M3: MUTANG + LAMS with shared enhanced pullback

## Goal

Run the **CLASP-faithful** stack: one ambient \(M\) (MDS from BLOSUM62 or
CLASP/AMP-BLOSUM) for both proposal generation and combination filtering.
Compare to Euclidean baseline (M0) and to the isolated arms (M1, M2) to
attribute effects.

## Method

1. Same parents; \(M \in \{I,\, M_{62},\, M_{\mathrm{CLASP}}\}\) from plan A (MDS).  
2. For each non-identity \(M\):
   - MUTANG from \(\mathrm{SVD}(\tilde J)\) with \(\tilde J=(I_L\otimes\sqrt{M})J\)  
   - LAMS from the same whitened tangent / pullback  
3. Calibrate thresholds so comparisons to M0 are fair (matched median MUTANG
   pool size before LAMS; same LAMS similarity threshold unless a retune is
   explicitly studied).  
4. Attribution (per parent × \(M\)):
   - \(\Delta\) vs M0 explained by pool change (compare to M1 outputs)  
   - \(\Delta\) vs M0 explained by filter change (compare to M2 outputs)  
   - Residual = interaction (new proposals × new angles)

## Metrics

Reuse D and G metrics on the **joint** outputs, plus:

| Metric | Purpose |
| --- | --- |
| Jaccard(retained_M3, retained_M0) | End-to-end candidate shift |
| Jaccard(retained_M3, retained_M1_then_euclid_LAMS) | If available: pool-only path |
| Conservative enrichment of final set | Does shared geometry help? |
| Emptiness rate (no multi-mutants retained) | Failure mode |

## Success / readout

- Prefer the source (\(M_{62}\) vs \(M_{\mathrm{CLASP}}\)) that improves conservative
  character of the **final** LAMS-retained set without collapsing yield.  
- State whether M3 ≈ M1 (metric matters only at MUTANG) or M3 needs shared
  LAMS geometry.

## Deliverables

- End-to-end comparison table (M0 / M1 / M2 / M3) for both MDS metrics  
- Default recommendation for CLASP ambient wiring  
- Notes for later \(\Delta\mathrm{cleav}<0\) gate (brief §5), not applied here  

## Dependencies

- Plans A, D, G  
- Shared whitening path used consistently for SVD and LAMS pullback
