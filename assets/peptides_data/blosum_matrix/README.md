# Substitution matrices (NCBI/BLAST text format)

Alphabet for all files: `ARNDCQEGHILKMFPSTWYV` (NCBI order, 20 standard amino acids).

This directory is kept in git (see `.gitkeep`). Matrix `.txt` files below are
versioned data artifacts used by ambient-metric analyses.

## Files

### `BLOSUM62.txt`

Standard NCBI/BLAST **BLOSUM62** substitution matrix.

| | |
| --- | --- |
| **Source** | NCBI BLAST matrices (`BLOSUM62`), loaded via BioPython `Bio.Align.substitution_matrices.load("BLOSUM62")` and subset to the 20 standard amino acids |
| **Canonical download** | [https://ftp.ncbi.nih.gov/blast/matrices/BLOSUM62](https://ftp.ncbi.nih.gov/blast/matrices/BLOSUM62) |
| **License / reuse** | NCBI distributes these matrices as U.S. Government Work with no restriction on use or reproduction; cite the source/authors in derived work |
| **Citation** | Henikoff S, Henikoff JG. Amino acid substitution matrices from protein blocks. *PNAS* 1992;89:10915–10919 |
| **Note** | Generic protein BLOSUM; **not** computed from AMP / DBAASP data |

### `DBAASP_10-50aa_BLOSUM62.txt`

DBAASP-derived AMP substitution matrix (integer half-bits), NCBI text format.

| | |
| --- | --- |
| **Corpus** | Peptides from [DBAASP](https://dbaasp.org) with length **10–50 aa** (filename); header reports `blocks_seen=2368`, `blocks_kept=2368` |
| **Construction** | Built with the [`blosum-matrix`](../../tools/blosum_matrix) package: `counting=plain`, `clustering=n/a`, labeled identity `0.62`, `unit=half_bits` (see file header) |
| **Note** | This is an **AMP-corpus** score matrix, not classical homologous-block NCBI BLOSUM62. Prefer documenting it as DBAASP 10–50 aa derived, not as a drop-in BLOSUM62 replacement |

Matrix construction code: [`assets/tools/blosum_matrix`](../../tools/blosum_matrix) (`blosum-matrix` package).
Regeneration scripts: [`assets/scripts/blosum_matrix`](../../scripts/blosum_matrix).
