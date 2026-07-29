# B — Source matrix deep-dive: BLOSUM62 vs CLASP (AMP-BLOSUM)

Optional companion to the main factorial in `README.md` (arms M1–M3 already
compare both sources end-to-end). Use this when you need a residue-pair-level
explanation of *where* the two \(S\) matrices disagree.

## Question

Does an AMP-specific substitution matrix improve on generic BLOSUM62 as the
ambient prior — the **only open comparison** called out in brief §4?

## Background

Base PepCompass uses Euclidean geometry: every amino-acid swap is equally far.
BLOSUM62 encodes general-protein evolutionary similarity. AMP-BLOSUM
(`data/blosum_matrix/AMPBLOSUM62_float.txt`) is built from equal-length DBAASP
blocks (Henikoff clustering at 0.62). Designed / AMPSphere sequences are
excluded by construction (brief §8).

## Hypotheses

1. Both BLOSUM matrices make conservative swaps (I↔L, K↔R) short and drastic
   ones (I↔P, K↔D) long; Euclidean cannot.
2. AMP-BLOSUM disagrees with BLOSUM62 mainly on pairs that matter for AMP
   chemistry (cationic amphipathic constraints: K/R/H vs hydrophobics,
   Pro/Gly vs helix formers).
3. Relative to BLOSUM62, AMP-BLOSUM shortens swaps enriched in natural AMP
   alignments and lengthens swaps that are common in globular proteins but rare
   in AMPs.

## Method

1. Fix the PD map chosen in plan **A** (or report both if undecided).
2. Build:
   - \(M_I = I\)
   - \(M_{62}\) from NCBI BLOSUM62
   - \(M_{\mathrm{AMP}}\) from AMP-BLOSUM float
3. Global AA×AA analysis (no peptides yet):
   - Rank all 190 unordered pairs by \(d_M^2\).
   - Diff ranks: \(\mathrm{rank}_{62}-\mathrm{rank}_{\mathrm{AMP}}\); list top disagreements.
4. Optional physicochemical anchors: correlate \(d_M^2\) with Grantham /
   Miyata / volume+charge distances (sanity, not ground truth).
5. Peptide-contextual check (light): for a panel of natural AMP parents
   (DBAASP / CAMP subset, length 8–25), score all 19 single mutants per
   position under each \(M\); compare which replacements are in the shortest
   quintile.

## Metrics

| Metric | What it answers |
| --- | --- |
| Separation ratio \(d^2(\mathrm{I},\mathrm{P})/d^2(\mathrm{I},\mathrm{L})\) | Conservative vs drastic contrast |
| Spearman(\(d_{62}^2\), \(d_{\mathrm{AMP}}^2\)) | Global agreement |
| Top-20 disagreement pairs + biochemical annotation | Where AMP prior differs |
| Per-parent: fraction of shortest-quintile mutants shared by 62 vs AMP | Local prior shift |

## Success / decision criteria

- If AMP vs 62 Spearman \(> 0.98\) and disagreement pairs are noise → **generic
  BLOSUM62 is enough**; AMP matrix is optional documentation, not a lever.
- If disagreements concentrate on AMP-relevant chemistry **and** MUTANG later
  (plan D) shows different short-proposal composition → keep AMP as default,
  62 as ablation arm (matches brief §6 step 3).
- Euclidean remains the negative-control geometry in all downstream plans.

## Deliverables

- Table of canonical pairs under \(I\), \(M_{62}\), \(M_{\mathrm{AMP}}\).
- Disagreement list (AMP vs 62) with short commentary.
- Default source matrix recommendation for CLASP ambient prior.

## Dependencies

- Plan **A** recommendation (or dual-track).
- BLOSUM files under `data/blosum_matrix/`.
- Optional: small curated parent list (can start with POC parent `GLLKRIKTLL`
  plus 10–20 DBAASP sequences).

## Notes

- Prefer float AMP scores; integer rounding zeros many off-diagonals
  (`data/blosum_matrix/README.md`).
- Do not rebuild AMP-BLOSUM from designed libraries or AMPSphere.
