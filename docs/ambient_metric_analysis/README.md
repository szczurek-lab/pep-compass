# Ambient metric analysis — experimental design

Computational ablations of the **enhanced ambient pullback metric** on
**MUTANG**, **LAMS**, and **MUTANG+LAMS**. Cleavage potential
\(\Phi_{\mathrm{cleav}}\) is out of scope here unless noted.

**PD construction:** classical MDS from BLOSUM scores only (brief §4).  
No matrix / elementwise exponential maps.

## Research questions

1. How does an enhanced pullback metric change **MUTANG** proposal sets vs Euclidean?
2. How does it change **LAMS** combination retention / angles — holding the proposal
   pool fixed, and jointly with metric-MUTANG?
3. Do **BLOSUM62** and the **CLASP (AMP-BLOSUM) metric**, built the same MDS way,
   disagree in ways that matter for MUTANG and/or LAMS?

## Factorial design

### A. Score source \(S\)

| Level | Matrix | Path |
| --- | --- | --- |
| `blosum62` | Generic NCBI BLOSUM62 | `data/blosum_matrix/BLOSUM62.txt` |
| `clasp` | AMP-BLOSUM (DBAASP / CLASP prior) | `data/blosum_matrix/AMPBLOSUM62_float.txt` (prefer float) |

Baseline geometry (no \(S\)): Euclidean \(M = I\).

### B. Positive-definite map \(S \to M\) (MDS only)

For both sources, the **same** construction (brief §4):

\[
D^2(a,b)=s_{\max}-S_{ab},\quad
J=I-\tfrac1A\mathbf{1}\mathbf{1}^\top,\quad
B=-\tfrac12 J D^2 J,\quad
M=B^{+}+\delta I.
\]

Then \(\|e_b-e_a\|_M^2 = D^2(a,b)\) (up to \(\delta\)). Similar residues → short moves.

Knobs: \(\delta\) (PD floor); optional overall scale \(\lambda\) on \(D^2\) or \(M\)
(ranks invariant — only matters for absolute path budgets).

Sanity-check \(\lambda_{\min}(M)\) and \(\kappa(M)\) before whitening.

Details: [`A_pd_construction.md`](A_pd_construction.md).

### C. Method arms (where \(M\) is inserted)

Enhanced pullback means ambient inner product \(G = I_L \otimes M\) via Jacobian
whitening \(\tilde J = (I_L \otimes \sqrt{M})\, J\) (pad token left as identity),
then the usual SVD / \(J_h^+\) stack.

| Arm | MUTANG pool | LAMS filter | Isolates |
| --- | --- | --- | --- |
| **M0** Euclidean baseline | \(\mathrm{SVD}(J)\) | Euclidean LAMS (\(M=I\)) | Reference |
| **M1** MUTANG-only | \(\mathrm{SVD}(\tilde J)\) | **off** (raw / capped product, or no combo filter) | Proposal effect of \(M\) |
| **M2** LAMS-only | \(\mathrm{SVD}(J)\) Euclidean pool **frozen** | LAMS with pullback under \(M\) | Filter/angle effect of \(M\) |
| **M3** MUTANG+LAMS | \(\mathrm{SVD}(\tilde J)\) | LAMS with same \(M\) | Shared geometry (CLASP-faithful) |

Notes:

- LAMS always consumes a MUTANG map; **LAMS-only** means the *pool* is Euclidean
  while *angles* use \(M\).
- CLASP combo rule later adds \(\forall i:\,\Delta\mathrm{cleav},i<0\); **not** in
  this metric-only analysis unless tagged as a follow-up.
- MOVE / LPBEBO / LE-BO GP are out of scope (not ambient-\(M\) consumers).

Arm protocols: [`D_mutang.md`](D_mutang.md), [`G_lams.md`](G_lams.md),
[`M_joint.md`](M_joint.md).

## What to measure

### Shared (per parent × condition)

- Parent panel: ≥30 natural AMPs (length 8–25); include POC parent `GLLKRIKTLL`.
- Calibrate `token_threshold` (or top-\(k\)) so median proposal counts are
  matched across \(M\) when comparing sets.

### MUTANG (M1 vs M0)

- Jaccard / size of single-position proposal sets  
- Mean / median \(d_M^2\) and mean BLOSUM score of proposed substitutions  
- Rank correlation within intersection  
- Pool size vs threshold curves  

### LAMS (M2 vs M0; also M3)

- Distribution of pairwise pullback cosines \(x_{ij}\)  
- Retention rate at the production LAMS threshold (and a small threshold sweep)  
- Overlap of retained multi-mutants  
- Mean conservatism of retained edits (BLOSUM / Grantham)

### Joint (M3 vs M1 vs M2 vs M0)

- Attribute change: “pool shift only” vs “filter shift only” vs “both”  
- Which score source maximizes conservative enrichment without emptying the
  retained set  

## Run matrix (practical)

1. **Build \(M\)** for `blosum62` and `clasp` via MDS; sanity PD / pair checks (A).  
2. **MUTANG arm** (M1) vs Euclidean.  
3. **LAMS-only arm** (M2): same \(M\)s, Euclidean pool fixed.  
4. **Joint arm** (M3): shared whitening for MUTANG and LAMS.  
5. Attribute end-to-end differences to pool vs filter vs both.

## File index

| File | Content |
| --- | --- |
| [`A_pd_construction.md`](A_pd_construction.md) | MDS \(S\to M\) for BLOSUM62 and CLASP |
| [`D_mutang.md`](D_mutang.md) | Arm M1 — MUTANG under enhanced pullback |
| [`G_lams.md`](G_lams.md) | Arm M2 — LAMS under enhanced pullback (frozen Euclidean pool) |
| [`M_joint.md`](M_joint.md) | Arm M3 — MUTANG+LAMS shared geometry |
| [`B_source_matrix.md`](B_source_matrix.md) | BLOSUM62 vs CLASP disagreement (optional deep-dive) |
| [`C_peptidase_gram.md`](C_peptidase_gram.md) | Pep-Gram keep/drop (deferred; default \(\lambda_{\mathrm{pep}}=0\)) |
| [`F_cleavage_interaction.md`](F_cleavage_interaction.md) | Follow-up with \(\Phi_{\mathrm{cleav}}\) |
| [`H_hyperparameters.md`](H_hyperparameters.md) | \(\delta\), \(\lambda\) (scale only) |
| [`I_local_search_proxy.md`](I_local_search_proxy.md) | Later LE-BO/SORBES proxy |
| [`E_insertion_mode.md`](E_insertion_mode.md) | Reweight vs whiten (only if M1 suggests equivalence) |

## Defaults / conventions

- Alphabet: `ACDEFGHIKLMNPQRSTVWY` (PepCompass order).  
- Prefer AMP float matrix; integer AMPBLOSUM zeros many off-diagonals.  
- No AMPSphere for \(S\) (brief §8).  
- No peptidase Gram in the main factorial (\(\lambda_{\mathrm{pep}}=0\)); see C.  
- No \(\mathrm{expm}\) / entrywise \(\exp\) in the analysis.  
- \(\lambda\) is a metric scale knob; do not confuse with \(\gamma\) (stability potential).  
- Report ranks (Spearman / Kendall \(\tau\)), not absolute distances, when comparing geometries.

## Out of scope (this round)

- Wet-lab C1–C3  
- PoGS / full LE-BO production wiring  
- MOVE, LPBEBO, MAP4 kernel  
- Exponential PD maps / temperature sweeps  
- Building \(M\) from designed peptides or AMPSphere  
