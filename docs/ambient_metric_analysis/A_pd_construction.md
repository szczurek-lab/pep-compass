# A — MDS metric from BLOSUM62 and CLASP (AMP-BLOSUM)

## Goal

Build a PD residue metric \(M\) from each score source **without** exponential
maps, using the brief §4 classical MDS construction. Sanity-check both \(M\)s
before MUTANG / LAMS arms.

## Construction (same for both sources)

Given symmetric score matrix \(S\) (reorder to PepCompass alphabet, symmetrize
\(\tfrac12(S+S^\top)\)):

\[
D^2(a,b)=s_{\max}-S_{ab},\quad
J=I-\tfrac1A\mathbf{1}\mathbf{1}^\top,\quad
B=-\tfrac12 J D^2 J,\quad
M=B^{+}+\delta I.
\]

- \(s_{\max} = \max_{a,b} S_{ab}\) (typically on the diagonal).  
- \(B^{+}\): PSD pseudoinverse — keep non-negative eigenspace of \(B\), invert
  those eigenvalues, zero the rest (classical MDS when \(D^2\) is not perfectly
  Euclidean).  
- \(\delta I\): numerical floor so \(M\) is strictly PD for \(\sqrt{M}\) whitening.

**Property:** \(\|e_b-e_a\|_M^2 = M_{aa}+M_{bb}-2M_{ab} = D^2(a,b)\) up to the
\(\delta\) contribution. High BLOSUM similarity ⇒ short substitution.

**Scale \(\lambda\):** replacing \(D^2\) by \(\lambda D^2\) (or \(M\) by \(\lambda M\))
does **not** change pairwise distance **ranks**; only absolute ambient lengths /
path budgets. Default \(\lambda=1\) for ranking studies.

## Score sources

| ID | \(S\) | Path |
| --- | --- | --- |
| `blosum62` | NCBI BLOSUM62 | `data/blosum_matrix/BLOSUM62.txt` |
| `clasp` | AMP-BLOSUM float | `data/blosum_matrix/AMPBLOSUM62_float.txt` |

## Sanity metrics (no decoder)

For each \(M\):

1. \(\lambda_{\min}(M)>0\), \(\kappa(M)=\lambda_{\max}/\lambda_{\min}\) — whitening safety  
2. Check \(d_M^2 \approx D^2_{\mathrm{ref}}\) (max absolute / relative error after \(\delta\))  
3. Canonical pairs: I↔L, I↔V short; I↔P, K↔D long; K↔R short  
4. Spearman of upper-triangle \(d^2\) between `blosum62` and `clasp` MDS metrics  

## Defaults

| Knob | Default |
| --- | --- |
| \(\delta\) | \(10^{-6}\) (spot-check \(10^{-8}\), \(10^{-3}\)) |
| \(\lambda\) | \(1\) |

## Deliverables

- Two frozen matrices \(M_{62}\), \(M_{\mathrm{CLASP}}\) (+ metadata: alphabet, \(\delta\), \(s_{\max}\))  
- Short sanity table (eigs, pair distances, cross-source Spearman)  
- Inputs for arms D / G / M_joint  

## Out of scope

- \(\mathrm{expm}\), entrywise \(\exp\), temperature / \(\alpha\) sweeps  
- Peptidase Gram (\(\lambda_{\mathrm{pep}}=0\); see `C_peptidase_gram.md`)

## Dependencies

- BLOSUM files under `data/blosum_matrix/`  
- Alphabet `ACDEFGHIKLMNPQRSTVWY`
