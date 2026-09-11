# blosum-matrix

Canonical Henikoff & Henikoff (1992) BLOSUM construction from pre-aligned protein
blocks. Produces an NCBI/BLAST text substitution matrix (20 standard amino
acids), readable by `Bio.Align.substitution_matrices.read`.

## Install

From this directory:

```bash
uv sync
```

Use the CLI via `uv run`:

```bash
uv run blosum-matrix build \
    --blocks-dir path/to/blocks --glob "*.fasta" \
    --identity 0.62 \
    --out-matrix BLOSUM62.txt
```

### Optional CD-HIT backend (GPL-2.0)

The default clustering backend is `exact` (pure Python; canonical; no GPL
deps). To use CD-HIT instead, install the optional extra and ensure the
`cd-hit` executable is on `PATH`:

```bash
uv sync --extra cdhit
```

Note: `py-cdhit` and `cd-hit` are licensed **GPL-2.0-only**. Installing the
`cdhit` extra and running `--clustering cdhit` brings GPL-2.0 components into
your environment. The default backend keeps the package MIT-clean.

## Regenerating checked-in matrices

Data-generation scripts (repo root):

```bash
cd assets/tools/blosum_matrix && uv sync
uv run python ../../scripts/blosum_matrix/generate_all.py
```

Or individually:

```bash
uv run python ../../scripts/blosum_matrix/export_blosum62.py
uv run python ../../scripts/blosum_matrix/generate_ampblosum62.py
```

Defaults write to `results/blosum_matrix/`.

## Inputs

- One or more **pre-aligned ungapped blocks** (BLOCKS-style). Aligned FASTA or
  Stockholm format; each file is **one block** and all sequences in a file must
  have equal width.
- The tool does **no** re-alignment, no per-cluster MSA, no global single-MSA
  simplification. Supplying unaligned sequences is rejected.
- Residues outside the standard 20 (`ARNDCQEGHILKMFPSTWYV`) and the gap
  character `-` are silently skipped per pair (do not fail the block); the CLI
  summary reports the number of skipped residues.

## Procedure (canonical)

1. Cluster within each block at threshold *T* (e.g. `0.62` → BLOSUM62).
2. Skip blocks with `< 2` clusters.
3. For each column, for every unordered cluster pair `(C_i, C_j)` and every
   sequence pair `s ∈ C_i, t ∈ C_j`, add `1/(|C_i| * |C_j|)` to the unordered
   amino-acid count `F[a, b]`. Same-cluster pairs are not counted.
4. Compute the canonical marginal `p_a = q_aa + 0.5 * sum_{b != a} q_ab`.
5. Half-bit log-odds, integer-rounded by default:
   - `s(a, a) = log2(q_aa / p_a^2) / u`
   - `s(a, b) = log2(q_ab / (2 * p_a * p_b)) / u`   for `a != b`
   - `u = 0.5` (half-bits, NCBI/BLAST convention) or `u = 1.0` (bits).
6. Emit a 20x20 NCBI-format text matrix on `ARNDCQEGHILKMFPSTWYV`. Ambiguity
   codes (`B`, `Z`, `X`, `*`) are not emitted.

## Backends

- `exact` (default): block-level pairwise identity computed as
  `matching_non_gap_columns / non_gap_in_both_columns`; connected components at
  threshold *T*; deterministic.
- `cdhit` (opt-in): clusters via `py-cdhit` calling the `cd-hit` binary on the
  block's **ungapped** residues. Requires `cd-hit` on `PATH`. **Block-level
  vs. full-length identity caveat**: CD-HIT clusters on full-length ungapped
  sequences, which equals block-level identity only when each file is one
  block.

## Failure modes (no silent fallback)

- Missing or failed `cd-hit` with `--clustering cdhit` → loud error with stderr.
- Any zero count after aggregation → loud error citing the offending `(a, b)`.
- Unequal width / `< 2` sequences / unaligned input → rejected before
  clustering.
