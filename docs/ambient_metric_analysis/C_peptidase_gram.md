# C — Peptidase Gram term: keep, drop, or reinterpret

## Question

Should the ambient metric include the POC’s MEROPS-derived peptidase Gram
\(M_{\mathrm{pep}}\), or is that term a category error relative to brief §4?

## Background

POC construction:
\[
M=\lambda_{\mathrm{blo}} M_{\mathrm{blo}}+\lambda_{\mathrm{pep}} M_{\mathrm{pep}}+\delta I,
\]
where each residue is embedded by concatenating log-preferences over all
protease×subsite slots and \(M_{\mathrm{pep}}\) is (normalized) \(EE^\top\).

Brief §2–4: a metric is a **symmetric** prior on moves; cleavage susceptibility
is **directional** and belongs in \(\Phi_{\mathrm{cleav}}\). Putting protease
specificity into \(M\) makes residue pairs that look similar *to proteases*
metrically close — including swaps that *increase* and *decrease* cleavage
alike, since \(d(a,b)=d(b,a)\).

## Hypotheses

1. \(M_{\mathrm{pep}}\) correlates with \(|\Delta\Phi_{\mathrm{cleav}}|\) for
   single substitutions more than BLOSUM distances do — i.e. it partially
   encodes cleavage magnitude, not just “neutral similarity.”
2. For a cleavage-hot bond, the shortest \(M_{\mathrm{pep}}\) replacements from
   a preferred P1/P1′ residue include both protective and equally bad
   alternatives (sign of \(\Delta\Phi_{\mathrm{cleav}}\) ignored).
3. \(\lambda_{\mathrm{pep}}>0\) changes MUTANG pools in a cleavage-looking way
   that should instead be achieved by \(\gamma\Phi_{\mathrm{cleav}}\) (plan F).

## Method

1. Build \(M_{\mathrm{pep}}\) from a fixed panel (POC default dataset index, plus
   one Gram-positive and one Gram-negative panel for robustness).
2. Build \(M_{\mathrm{blo}}\) (AMP and/or 62 from plan B) with \(\lambda_{\mathrm{pep}}=0\).
3. **Leakage test:** for many (parent, position, aa→aa′) singles:
   - \(d^2_{\mathrm{pep}}\), \(d^2_{\mathrm{blo}}\)
   - \(\Delta\Phi_{\mathrm{cleav}}\) and \(|\Delta\Phi_{\mathrm{cleav}}|\)
   - Rank correlation: \(d^2_{\mathrm{pep}}\) vs \(|\Delta\Phi|\); vs signed \(\Delta\Phi\)
     (signed should be ~0 if purely symmetric).
4. **Sign blindness:** at positions with high wild-type bond rate, take the
   \(k\) nearest residues under \(M_{\mathrm{pep}}\) and count how many raise vs
   lower \(\Phi_{\mathrm{cleav}}\).
5. **Ablation on pools:** MUTANG proposal overlap for
   \(\lambda_{\mathrm{pep}}\in\{0,0.5,1\}\) at fixed \(\lambda_{\mathrm{blo}}\) (preview of D).

## Metrics

| Metric | Expected if pep Gram is inappropriate in \(M\) |
| --- | --- |
| Spearman(\(d^2_{\mathrm{pep}}\), \(|\Delta\Phi|\)) | High (leakage of cleavage magnitude) |
| Spearman(\(d^2_{\mathrm{pep}}\), signed \(\Delta\Phi\)) | ~0 (symmetry) |
| Protective fraction among \(k\)-NN under \(M_{\mathrm{pep}}\) | ~50% (coin flip on direction) |
| Spearman(\(d^2_{\mathrm{blo}}\), \(|\Delta\Phi|\)) | Lower than pep (control) |

## Success / decision criteria

- If leakage is strong and nearest neighbors are direction-blind → **set
  \(\lambda_{\mathrm{pep}}=0\) by default**; document pep Gram as a rejected
  extension (or move the idea into the potential / filter, not the metric).
- If pep Gram mostly recapitulates BLOSUM (high agreement, weak \(|\Delta\Phi|\)
  correlation) → drop it as redundant.
- Only keep \(\lambda_{\mathrm{pep}}>0\) if you can state a *symmetric* prior story
  that does not substitute for \(\Phi_{\mathrm{cleav}}\) (e.g. “residues that
  proteases treat interchangeably are interchangeable for search step size”)
  **and** plan F still shows clear gains from \(\gamma\) on top.

## Deliverables

- Leakage / sign-blindness figures for one primary panel.
- Explicit default: \(\lambda_{\mathrm{pep}}=0\) or not, with one-paragraph rationale
  tied to brief §2 (“metric cannot represent reduce cleavage”).

## Dependencies

- Working \(\Phi_{\mathrm{cleav}}\) (or per-bond rate) evaluator on decoder
  simplex / one-hot peptides.
- Specificity tensors under `data/specificity_matrices/`.
- Plans A–B for \(M_{\mathrm{blo}}\).

## Notes

- Strain-conditional \(\rho_\pi(\sigma)\) belongs in \(\Phi(\sigma)\), not in a
  single global \(M_{\mathrm{pep}}\). A strain-specific pep metric would multiply
  the category-error risk.
