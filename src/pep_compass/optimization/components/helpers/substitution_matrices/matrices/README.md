# Substitution matrices (NCBI/BLAST text format)

Alphabet for all files: `ARNDCQEGHILKMFPSTWYV` (NCBI order, 20 standard amino acids).

Matrix construction: [`assets/tools/blosum_matrix`](../../../../../../assets/tools/blosum_matrix) (`blosum-matrix` package).
Regeneration scripts: [`assets/scripts/blosum_matrix`](../../../../../../assets/scripts/blosum_matrix).

## Files

| File | Provenance |
| --- | --- |
| `BLOSUM62.txt` | Official NCBI BLOSUM62 shipped with BioPython (`Bio.Align.substitution_matrices.load("BLOSUM62")`), subset to the 20 standard amino acids. **Not** computed from AMP data. |
| `AMPBLOSUM62.txt` | Built with `blosum-matrix` (Henikoff & Henikoff 1992): exact block-level clustering at identity `0.62`, half-bit log-odds, integer rounding. Inputs: `data/hydramp/dbaasp/blosum_blocks/len_*.fasta` (one equal-length DBAASP peptide set per file = one ungapped block). |
| `AMPBLOSUM62_float.txt` | Same build as `AMPBLOSUM62.txt` with `--no-round` (float half-bits). Prefer this for metric / ambient geometry work; integer AMP scores often round many off-diagonals to zero. |

See also the comment header inside each `.txt` file for the exact tool parameters and BioPython version used at generation time.
