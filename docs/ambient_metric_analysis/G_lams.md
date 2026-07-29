# G — Arm M2: LAMS under enhanced pullback (frozen Euclidean MUTANG)

## Goal

Isolate the effect of ambient \(M\) on **LAMS** combination filtering: keep the
**MUTANG proposal pool Euclidean**, but compute LAMS pullback directions /
cosines under the enhanced metric.

## Why freeze the pool

If MUTANG and LAMS both see \(M\), pool shift and angle shift are confounded.
M2 answers: *given the same candidates, does \(M\)-geometry change which
multi-mutants LAMS keeps?*

## Method

1. Same parent panel as D.  
2. Build MUTANG map once with \(\mathrm{SVD}(J)\) (Euclidean); freeze it.  
3. For \(M \in \{I,\, M_{62},\, M_{\mathrm{CLASP}}\}\) (MDS from A):
   - Build tangent / pullback from **whitened** \(\tilde J\) (or equivalent
     \(M\)-inner-product pullback) while still enumerating combinations from the
     **frozen Euclidean** mutation map.  
   - Run `LamsFilter` / `LamsAnchorSimilarityPotential` at the production
     similarity threshold; also sweep threshold lightly.  
4. Cap Cartesian products with the same `maximum_candidates` policy as
   production.

Implementation note: production LAMS rebuilds SVD from the current Jacobian
inside the filter. For this arm, inject whitened geometry into that SVD /
`SubRiemannianTangentSpace` (or POC-equivalent) without regenerating the MUTANG
residue lists from the whitened \(U\).

## Metrics

| Metric | Purpose |
| --- | --- |
| Retention rate (fraction of combinations with min cosine ≥ thr) | Filter tightness |
| Distribution of pairwise \(x_{ij}\) among mutated pairs | Angle shift |
| Jaccard of retained multi-mutant sets vs Euclidean LAMS | Set change |
| Mean BLOSUM / Grantham of retained edits | Conservatism of survivors |
| Fraction single-mutants retained | Sanity (singles should still pass) |

## Success / readout

- If retention and retained sets barely move → LAMS is insensitive to residue
  \(M\) at fixed pool; metric impact is mostly MUTANG (arm M1).  
- CLASP vs BLOSUM62: do they disagree more on **angles** than on single-mutant
  lengths?

## Deliverables

- Retention tables: source × LAMS threshold  
- Recommendation: whether LAMS must share \(M\) with MUTANG or can stay Euclidean  

## Dependencies

- Plan A shortlist; frozen Euclidean maps from the same Jacobians as D  
- `LamsFilter` / `LamsAnchorSimilarityPotential` in
  `src/pep_compass/local_enumeration/mutation/`
