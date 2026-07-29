# H — Scale and ridge sensitivity (MDS)

MDS geometry is fixed by \(S\) up to rank-preserving scale. This file covers
the remaining knobs \(\delta\) and \(\lambda\).

## Question

Do \(\delta\) or \(\lambda\) change substitution **ranks** / MUTANG sets, or only
absolute lengths?

## Hypotheses

1. Positive \(\lambda\) on \(D^2\) or \(M\) leaves pairwise ranks invariant.  
2. Tiny \(\delta\) (\(\lesssim 10^{-4}\)) is invisible to ranks; large \(\delta\)
   Euclideanizes \(M\) toward \(I\).

## Method

1. Fix \(M_{62}\) and \(M_{\mathrm{CLASP}}\) from plan A.  
2. Grid: \(\lambda\in\{0.25,1,4\}\), \(\delta\in\{10^{-8},10^{-6},10^{-3},10^{-1}\}\).  
3. Kendall \(\tau\) of the 190-pair distance ranking vs default
   \((\lambda=1,\delta=10^{-6})\).  
4. On ~10 parents: MUTANG top-set Jaccard vs default (arm M1 settings).

## Deliverables

- Confirm ranks stable in the default region; document when \(\delta\) starts to
  matter.  
- Freeze \((\lambda,\delta)\) for joint runs.

## Dependencies

- Plans A, D.
