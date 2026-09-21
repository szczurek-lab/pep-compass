# Serum proteolysis calibration — data audit

Phase-3 audit of the calibration described in
`tmp/clasp/serum_proteolysis_calibration_spec_PL_updated.md` and
`tmp/clasp/serum_data_acquisition_section.md`.

## Scope

The audit determines for which proteases the data exist to test the hypothesis
that the MEROPS window score predicts quantitative cleavage kinetics. It does not
test that hypothesis and fits no model.

Observations, findings and open questions are recorded in [report.md](report/initial_datasets/analiza_0_01.md).
Intermediate tables are written to [results/](results/). The methods are defined
in the markdown cells of the notebooks that implement them.

## Methodology

This section is self-contained: it states what is being tested, where every input
came from and under what licence, what each input is and is not evidence of, how
every metric is defined, how each test is constructed, and what the results
license. It is written so that the numbers can be checked against the sources and
the procedures against the code.

Findings are not deferred to the end. Each subsection closes with a
**Findings and decisions** block giving what the analysis returned and what was
decided as a consequence, because a methodological choice made without its result
in view cannot be judged.

### The chain of inference

The hypothesis is that a MEROPS specificity matrix, which is a tally of cleavages
reported in the literature, carries enough quantitative information to predict how
fast an arbitrary peptide is destroyed in human serum, and therefore how long it
keeps its antimicrobial activity. That claim is a chain, and each link is a
separate empirical question:

| # | Link | Object | Tested in | Falsified by |
| --- | --- | --- | --- | --- |
| 1 | a specificity matrix ranks the bonds of a peptide | $S^M_{\pi,b}$ | `2_02`, `2_05`, `3_03` | published cleavage sites scoring no better than chance |
| 2 | that ranking is monotone in measured catalytic efficiency | $S \to k_{cat}/K_M$ | `2_01`, `2_03`, `2_04`, `2_06` | within-protease rank correlation indistinguishable from zero on held-out clusters |
| 3 | per-bond intensities sum to a serum hazard | $\lambda_{\pi,b} \to \Lambda$ | `3_01` | modelled hazard failing to order measured half-lives |
| 4 | the hazard generates a cleavage process | Gillespie chain | `4_01`, `4_02` | — (a modelling choice, not an empirical claim) |
| 5 | the fragments have predictable potency | fragment MIC | `4_03` | products known to retain activity scoring worse than products known to lose it |
| 6 | fragment potencies aggregate to a mixture activity | $\psi = \sum_i C_i/\mathrm{MIC}_i$ | `4_03`, `4_02` | a digest known to abolish activity scoring more potent than the intact peptide |

A link can hold while the next fails, and in this analysis that is what happened.
The order above is the order in which the notebooks are numbered, and §6 states
which links survived.

### Data selection

Each source below is described as: what it physically is, what kind of information
it can and cannot support, how it was obtained, what was extracted, and what was
rejected. Where a source is licensed or was retrieved manually, that is stated,
because it determines whether the step is reproducible by a third party.

#### MEROPS specificity matrices

##### What the object physically is

The matrices come from the MEROPS `Substrate_search` release of 2023
(<https://www.ebi.ac.uk/merops/>), reorganised into `data/merops/` as 36 datasets.
Dataset ids 0–33 are the 34 APEX pathogen strains in the exact order of the APEX
output columns, so the dataset id equals the APEX column index; id **34** is the
full *Homo sapiens* peptidase set; id **35** is the human serum/plasma subset that
the CLASP specification designates as the default panel.

Each dataset holds `matrices_counts.npy` and `matrices_logprob.npy`, tensors of
shape `[N, 8, 20]`. Axis 1 is the subsite in the order
`P4, P3, P2, P1, P1', P2', P3', P4'`; axis 2 is the residue in the order
`ACDEFGHIKLMNPQRSTVWY`, identical to the APEX alphabet. A peptidase appears in a
dataset only if the organism has an ortholog **and** MEROPS records at least one
cleavage for it; peptidases with no cleavage data are absent rather than zero.

##### What kind of information this is, and where it is slippery

This is the single most important qualification in the whole analysis, and it
propagates into every result.

A MEROPS matrix is a **tally of cleavage events reported in the published
literature**. Each recorded cleavage contributes one count to the residue observed
at each of the eight subsites. There is no weighting by substrate abundance, no
weighting by rate, and no distinction between a cleavage observed once in a
synthetic reporter peptide and a cleavage that dominates a physiological
digest. All cleavage types are pooled — natural substrates and synthetic
substrates alike — and all protein sequences assigned to one MEROPS code are
pooled into one matrix, matching the code-level matrix the MEROPS website
displays.

Three consequences follow, and they are not cosmetic.

1. **The matrix is a frequency profile of what was studied, not of what happens.**
   A protease whose literature consists largely of fluorogenic reporter substrates
   designed around a known preference will return a matrix that restates that
   design. The specificity looks sharp because the substrate pool was selected to
   make it sharp.
2. **Counts are not effort-corrected.** A peptidase with 4 330 recorded cleavages
   and one with 24 are both a single row. The sampling floor of any specificity
   statistic therefore differs by two orders of magnitude between rows of the same
   panel, which is why an information-content excess is measured against a
   resampled null at the row's own count rather than compared across rows.
3. **Pooling across sequences assumes one code is one active site.** This is
   defensible — the `MER` accessions under a code are the same peptidase — but it
   means the matrix cannot represent allelic or isoform differences, and the
   source table does not retain the per-sequence accession, so the assumption
   cannot be relaxed later without re-extracting from MEROPS.

Nothing downstream can repair this. It is the reason the analysis reports where a
peptide is attacked with more confidence than how fast.

##### Which dataset supplies which protease

`best_merops_dataset(code, merops_root)` returns the dataset in which a code
carries the most recorded cleavages. For the serum model the panel is restricted
to human datasets, because the calibration is about human serum; the restriction
is asserted in `2_02` and `2_05`.

One exception is deliberate and is kept separate in the code. The published
cleavage bonds available for validation were measured with bacterial enzymes —
aureolysin `M04.009`, staphylococcal V8 `S01.269`, pseudomonal LasB `M04.005` —
so the site-recovery validation resolves matrices through an unrestricted lookup
while the serum model resolves through the restricted one. Mixing the two would
smuggle bacterial enzymes into a human serum panel; keeping them apart is what
allows the bacterial evidence to be used at all.

##### Smoothing

Subsite distributions are estimated with a Jeffreys prior, $\alpha = 0.5$, added to
every count before normalising. The value was verified numerically against the
distributed `matrices_logprob.npy` rather than taken on trust, and it is left
unchanged, because the audit measures the matrix the model actually consumes.

> **Discrepancy on record.** `data/merops/README.md` describes
> `matrices_logprob.npy` as *Laplace*-smoothed, which would be $\alpha = 1$. The
> numerical check gives $\alpha = 0.5$. The code in this analysis uses
> `MEROPS_SMOOTHING_ALPHA = 0.5`, consistent with the file rather than with its
> description.

##### The reference distribution

Subsite specificity is the divergence of the subsite distribution from a reference
distribution, and the choice of reference is a scientific claim, not a formality.

The reference used is the residue composition of the reviewed human proteome
(§ *Human reference proteome*). A uniform reference asserts that the null
expectation for a subsite is equal use of all twenty residues, which is false for
any real substrate pool: leucine occurs at $0.0996$ and tryptophan at $0.0121$ in
the human proteome, a spread of $8.2\times$. Against a uniform reference, a subsite
that merely reproduces the composition of its substrate pool is reported as
specific, and a constant $0.145$ bits is added to every value in the panel.

The score itself is therefore a log-odds against the proteome background, not a
log-probability, and is reported in nats. A negative window score means the
residues at that bond are **less** likely under the protease's profile than under
the background — the bond is disfavoured, not merely unobserved.

> **Findings and decisions.** The reference was changed from uniform to the human
> proteome partway through the analysis and every specificity figure was
> recomputed; the change is documented in `0_01`. The sampling floor of the
> information-content statistic turned out to be **non-monotonic** in the number of
> cleavages — it peaks near $n \approx 15$ under the Jeffreys prior — so a raw
> information content cannot be compared between proteases with different counts,
> and the excess over a resampled null is used instead.

#### BRENDA kinetics

##### What the object physically is, and the licence constraint

BRENDA (<https://www.brenda-enzymes.org/>) release **2026.1**, obtained as the
licensed bulk JSON export, `brenda_2026.1_json.tar.gz`, roughly 677 MB compressed.
The export is **not** retrievable by an automated request: it requires an accepted
licence and a manual download. `install_brenda_bulk_export.py` registers a manually
placed archive and `check_brenda_snapshot.py` verifies it, so the pipeline records
a snapshot it did not fetch.

The archive is parsed by streaming with `ijson` rather than loading, because the
decoded structure does not fit comfortably in memory.

##### What kind of information this is

BRENDA entries are curated transcriptions of values reported in papers, attached to
an EC number, an organism and a protein record. They are **not** a controlled
experiment: assay buffer, temperature, substrate construct and detection method
vary between entries and are not normalised. Two entries for the same enzyme and
substrate can differ by an order of magnitude for reasons that are real and
uncontrolled. This is why every correlation in §4 is a **rank** correlation
computed **within** a protease, never across proteases, and never on absolute
values.

##### The two extraction scopes, and why both are kept

Two extractions exist and both are retained.

- **Serum scope** — the 40 distinct EC numbers of the serum protease manifest.
  Receipt: 40 requested, 40 present in BRENDA, **3 803 entries**. Input to `0_03`
  and `2_01`.
- **Peptidase scope** — every EC number under `3.4.` for which BRENDA holds human
  kinetics, reached by mapping MEROPS codes to EC through UniProt. Receipt: **139**
  EC numbers, 139 present, **8 366 entries**. Selected with `--ec-mapping`. Input
  to `2_02` onward.

The wider mapping runs EC → UniProt → MEROPS, the opposite direction from the
manifest, which was assembled MEROPS → UniProt → EC. Their overlap is therefore a
cross-check and not a restatement.

Both are produced once, outside the notebooks:

```shell
# repository root
uv run assets/scripts/downloads/serum_proteolysis_calibration/ingest_brenda.py
uv run assets/scripts/downloads/serum_proteolysis_calibration/ingest_brenda.py --ec-mapping
```

Outputs land in
`data/serum_proteolysis_calibration/processed/kinetics/` as
`brenda_serum_proteases.parquet` and `brenda_all_peptidases.parquet`, each with a
coverage sidecar recording the EC numbers requested, the EC numbers present and the
per-EC entry counts, and an ingest receipt carrying the archive sha256.

##### Filters applied at extraction

Entries are restricted to *Homo sapiens* protein records at extraction time, to EC
numbers under `3.4.`, and to entries whose substrate field can be parsed into a
sequence. Values and units are preserved exactly as BRENDA reports them; no unit
harmonisation is performed at ingest, and the normalisation happens in the
notebooks where it can be inspected.

##### The kinetic label

The quantity the hypothesis is about is catalytic efficiency, $k_{cat}/K_M$.

Across the **expanded** peptidase scope BRENDA reports **22** window-resolved
entries directly on that scale. Where one publication reports $k_{cat}$ and $K_M$
for the same protein record and the same substrate, their ratio is the same
quantity; pairing on the key (protease, substrate, reference, protein record)
recovers a further **156** observations. These are flagged as derived in the
results and can be excluded.

The corresponding figure for the **serum** scope alone is **46** observations on an
absolute scale across the whole panel, which `0_09` records as insufficient to
calibrate even one protease. The gap between these two numbers is the reason the
scope was widened at all, and it is why the two figures must not be quoted
interchangeably: they count different panels.

The two parameter tables are joined on the **intersection** of their keys, with
`pd.concat([turnover, michaelis], axis=1, join="inner")`. The obvious alternative,
a `pivot_table` over the seven-column key with `dropna=False`, materialises the
full cross-product of the key levels and requests 194 GiB; it is not a stylistic
preference but a hard constraint.

##### Duplicate and ambiguous EC assignments

This is the second place where the data are structurally slippery, and it is
handled explicitly rather than averaged away.

An EC number names a *catalytic activity*, not a gene. Several paralogues can share
one EC number, and MEROPS assigns each paralogue its own code. A BRENDA entry
carrying that EC therefore attaches to **every** code sharing it, although exactly
one enzyme catalysed the measured reaction and the data do not say which.

Measured: **2 001 of 8 308** entries (24%) attach to more than one MEROPS code.

Three responses were considered and the third was taken.

1. *Drop ambiguous entries.* Rejected: it removes a quarter of the data and
   preferentially removes the well-studied families, which are the ones with
   paralogues.
2. *Divide the evidence between codes.* Rejected: it invents a weighting the data
   do not support and makes the count depend on how finely MEROPS happens to split
   a family.
3. *Attach the entry to every candidate code and carry the ambiguity as a
   column.* Taken. Every protease carries
   `shares_measurements_with_n_codes` through to `2_07_model_panel.csv`, and a
   result on a protease with a value above one is a result about the family, not
   about the gene.

The same reasoning applies to the reverse direction: `map_ec_to_merops.py` resolves
EC to MEROPS through **reviewed** human UniProt records only, requesting the fields
`accession,protein_name,gene_primary,ec,xref_merops` from
<https://rest.uniprot.org>. Unreviewed records are excluded because their EC
annotations are propagated automatically and would multiply the ambiguity without
adding evidence.

A curated override file, `config/curated_ec_to_merops.tsv`, holds **5** bacterial
assignments, each with its source. It is **opt-in** (`--curated`, default off), so
a bacterial enzyme can never enter the human panel by accident.

> **Findings and decisions.** Widening the scope moved the extraction from 3 803 to
> 8 366 entries, from 402 to **990** usable (protease, substrate) pairs, and from 27
> to **101** MEROPS codes. The cross-check in `0_10` found that the two mappings
> agree on all **42** shared assignments and that the wide extraction reproduces the
> narrow one row for row. Because widening changes what is being measured, the
> regression notebooks were duplicated rather than replaced: `2_02`/`2_03` run on
> the narrow scope and `2_05`/`2_06` on the expanded one, so the cost of the
> expansion is visible (§4).

#### The human serum protease manifest

`build_serum_manifest.py` and `map_uniprot_serum_proteases.py` assemble
`data/serum_proteolysis_calibration/manifests/serum_proteases.tsv`: **55** rows, of
which **54** carry `include_in_hydrolysis_model = true`, spanning **40** distinct EC
numbers. Columns record the MEROPS code, name and family, the number of MEROPS
cleavages, a serum category, the UniProt accession, gene name, Ensembl gene id, EC
number and the mapping status.

`include_in_hydrolysis_model` gates membership of the hydrolysis panel; it does not
gate the audit, so a protease excluded from the model is still counted in the
coverage tables and the reason for its exclusion stays visible.

#### Protease abundance in blood

Two Human Protein Atlas tables (<https://www.proteinatlas.org/>):

- <https://www.proteinatlas.org/download/tsv/blood_ms_concentration.tsv.zip>
- <https://www.proteinatlas.org/download/tsv/blood_immunoassay_concentration.tsv.zip>

Both give a blood concentration per gene; units are normalised to pg/L. The two
methods disagree, systematically and by large factors, because mass spectrometry
under-reports proteins that are hard to ionise or are present as complexes, while
immunoassays are antibody-dependent and can over-report.

The rule implemented in `0_07` is
`best_estimate_pg_per_l = max(ms_pg_per_l, immunoassay_pg_per_l)`, and the method
of origin is retained in `abundance_methods`. The rule is a deliberate bias toward
the higher estimate: a protease that one method says is abundant is treated as
abundant. The alternative — preferring MS as the more uniform method — would
systematically deplete exactly the plasma proteases the model is about.

Weights enter the model as $w_\pi = a_\pi / \mathrm{median}(a)$, the abundance
divided by the panel median. The division is a pure rescaling absorbed by the
intercept $\alpha$; it is done so that $\alpha$ has a comparable meaning between the
weighted and unweighted variants.

> **Findings and decisions.** **47** of the **57** panel proteases have an abundance
> estimate. The remaining ten can only enter the uniform-weighted variant, which is
> one reason both variants are carried through every downstream notebook rather
> than one being chosen.

#### Site-resolved cleavage sources

These sources give *where* a protease cuts, not how fast, and they are the evidence
for link 1 of the chain.

| Source | Accession or link | What it gave |
| --- | --- | --- |
| MMP PICS | PRIDE `PXD002265`, <https://www.ebi.ac.uk/pride/archive/projects/PXD002265> | proteome-derived cleavage sites for matrix metalloproteinases, audited in `0_05` |
| DPP4 qPISA | PRIDE `PXD042089` and PMC `PMC11612144` | quantitative subsite preference for DPP4, audited in `0_04`, used as the external check in `1_02` |
| PepLife2 | <https://webs.iiitd.edu.in/raghava/peplife2/> | peptide serum half-lives, audited in `0_06` |
| DRAMP stability | <https://dramp.cpu-bioinfor.org/> (`stability_amps.xlsx`) | stability annotations for antimicrobial peptides, audited in `0_06` |
| SABIO-RK | <https://sabiork.h-its.org/> | **not retrievable**; the directory is empty and the state is recorded as a blocker |

Two further sources named in the specification — the qMSP-MS tables and the
ADAMTS13 supplement — were not retrievable either. Together with **13** panel
proteases that have no EC number or an ambiguous UniProt mapping, these are
recorded in `0_08` as blockers with a reason, and never as zero counts. The
distinction matters: a zero says the evidence was sought and is absent, a blocker
says the evidence was not reachable.

#### Human reference proteome

`download_human_proteome.py` retrieves the reviewed *Homo sapiens* reference
proteome **UP000005640** from <https://rest.uniprot.org> and computes the residue
composition: **20 416** sequences, **11 414 601** residues, leucine at $0.0996$ and
tryptophan at $0.0121$. Output: `data/reference/human_proteome/composition.json`,
with the FASTA snapshot and a retrieval receipt beside it.

```shell
# repository root
uv run assets/scripts/downloads/serum_proteolysis_calibration/download_human_proteome.py
```

#### The biologist's control panel

`configs/clasp-controls.tsv` is the one source in the analysis that is not a
database dump: it is a hand-curated table of peptides whose proteolytic fate is
published, with the enzyme, the assay format, the cleaved bonds where they are
known, and a citation for each row.

Its structure encodes the evidence grade, and every statistic computed on it
respects that grading:

- `confidence` — `high`, `medium` or `inferred`;
- `verified_by` — who read the source (`claude-pdf`, `claude-web`, `paulina`,
  `unverified`);
- `cleavage_sites` — either explicit bonds (`R19|I20`), or a marker such as
  `unmapped`, `none_reported`, `unassigned`, or a product description;
- `notes` — states explicitly where a bond was **published as a bond** and where it
  was reconstructed by mapping a homologous peptide.

Every bond used in `3_03` is re-verified against the sequence before use: the P1 and
P1′ letters recorded in the table must be the residues actually present at those
indices, and the indices must be consecutive. A row that fails is dropped rather
than re-interpreted.

Principal citations, so the rows can be checked:

| Peptide and enzyme | DOI or accession |
| --- | --- |
| LL-37 with aureolysin and V8 | <https://doi.org/10.1128/AAC.48.12.4673-4679.2004> |
| EFK17, EFK17-W with aureolysin, V8, LasB, HNE | <https://doi.org/10.1128/AAC.00477-08> |
| LL-37 with *P. aeruginosa* elastase and gelatinase | <https://doi.org/10.1046/j.1365-2958.2002.03146.x> |
| LL-37 processed by KLK5 and KLK7 | <https://doi.org/10.1096/fj.06-6075com> |
| LL-37 with EHEC and EPEC OmpT | <https://doi.org/10.1128/IAI.05674-11> |
| catestatin and cateslytin with *S. aureus* supernatants | <https://doi.org/10.1371/journal.pone.0068993> |
| hBD-2 and hBD-3 with cathepsins B, L and S | <https://doi.org/10.4049/jimmunol.171.2.931> |
| oncocin Onc18 in serum | <https://doi.org/10.1371/journal.pone.0178943> |
| CAM and CAM-W stability panel | PII `S0006291X14014594` (BBRC 451:650-655, 2014) |
| GKY25 with aureolysin | PMC `PMC3101415` |

One row is recorded as a **counterexample** rather than as evidence: cateslytin
`bCTL` retains both of the bonds cleaved in its parent `bCAT`, with identical
P4–P4′ windows, and is nevertheless not degraded by any of four *S. aureus*
strains. The authors attribute the resistance to 33% arginine content and an
aggregated antiparallel β-sheet. A window-only model has no mechanism for this, so
the row is used as a bound on what the model can explain, never as a test it
passes.

`data/peptides_data/peptides_mic/controls_mic.csv` is a second, independent label
set: **10** peptides with a binary `serum_fate` of `stable` or `labile`, from two
sources (OmegAMP designs and a palaeogenomics supplement), with an `active` flag.
Three of the ten are recorded as inactive, so their serum fate was not necessarily
established by the same assay as the rest — a caveat that is stated wherever the
set is used.

#### Peptide corpora and MIC predictions

Antimicrobial potency is predicted with the APEX ensemble
(`pep_compass...oracles.strategies.apex.predictor.APEXPredictor`), which returns a
MIC for each of **34** strains. The reference strain used throughout is
*A. baumannii* ATCC 19606, the first output column; where a published result was
measured on other species, the comparison is repeated on all 34 columns rather than
argued about (`4_03`).

- `data/peptides_data/peptides_mic/peptides_apex.csv` — predictions for the full
  peptide corpus, the `dataset` column identifying the sub-corpus.
- `data/peptides_data/peptides_mic/dbaasp_fragments_mic.csv` — **211 040**
  precomputed fragment predictions, lengths 1 to 25.

Fragments longer than 25 residues are not in the table, so notebooks that need
them call APEX directly and cache the result. `4_02` precomputes every contiguous
subsequence of each probe peptide once, which turns the simulation into table
lookups.

> **Findings and decisions.** The MIC predictor is used far outside its training
> domain when it is asked about four-residue fragments, and it does not refuse: it
> returns a finite MIC only modestly worse than the parent's. This is the direct
> cause of the mixture-rule failure in §6, and the reason a minimum-length guard
> exists at all.

### Source audits: what each source can support, and how they differ

The `0_` notebooks are one audit per source, followed by a synthesis. Their purpose
is not descriptive. Before any model is fitted they answer three questions that
decide the shape of the whole analysis: **does the data itself carry information**,
**which proteases have to be dropped for want of data**, and **what kind of claim
each source can support**. A source that cannot support a claim is not a weaker
version of one that can; it answers a different question, and adding them together
would select proteases by whichever question is easiest to satisfy.

#### The axis that separates the sources

The sources differ along four axes at once, and confusing any two of them produces
a false result.

| Axis | The two ends | Consequence if ignored |
| --- | --- | --- |
| **Curation** | hand-curated by a database team from published papers (MEROPS, BRENDA) against raw instrument output deposited by one laboratory (PICS, qPISA) | curated sources inherit the biases of what was published; deposited sources inherit the biases of one protocol |
| **Assay format** | fluorogenic reporter peptide, proteome-derived peptide library, whole serum incubation | a reporter measures a designed bond; a library measures whichever bonds exist in the proteome; serum measures everything at once |
| **Measurement type** | absolute rate ($k_{cat}/K_M$), relative abundance change, bare occurrence of a cleavage | only the first is on a scale; the other two constrain the **shape** of a relation and can never calibrate it |
| **Site resolution** | scissile bond explicit, bond fixed by geometry, bond unknown | a score computed on an assumed bond and then correlated with that assumption is circular |

`0_09` therefore reports the decision on **three independent axes** — specificity
evidence, quantitative kinetics, control-panel relevance — and never as one score.

#### `0_01` — does the MEROPS matrix carry information at all

The first question is whether the matrices are worth scoring. For each of the 55
serum-panel proteases the notebook measures the divergence of each subsite from the
proteome background, the divergence a matrix of the **same count** would show by
chance, their difference, and the stability of the resulting bond ranking under
resampling of the cleavages.

The counts are wildly unequal: from **1** recorded cleavage to **3 417**, median
**20**. That alone dictates the method — an absolute divergence cannot be compared
between rows, so the excess over a count-matched null is used.

**Findings and decisions.**

| Matrix verdict | Proteases |
| --- | --- |
| `well_determined` | 11 |
| `usable` | 4 |
| `weak` | 35 |
| `uninformative` | 5 |

Only 15 of 55 matrices are better than weak. The five `uninformative` ones are
rejected here and not by any later statistic: coagulation factor XIIIa, for example,
has **one** recorded cleavage, so its matrix is the Jeffreys prior with a single
observation added and its "specificity" is the prior.

Stability is measured by resampling the cleavages and recomputing the bond ranking:
the median protease reproduces its ranking at **0.726**, and the tenth percentile of
that distribution is **0.568**. Half the panel therefore produces a bond order that
a different draw of the same literature would substantially rearrange. This is the
quantitative form of the warning in *What kind of information this is*: the matrices
are stable enough to rank bonds on average and not stable enough to trust
protease by protease.

A separate result was that the panel is **not** redundant: clustering the scores
produced by the matrices gives **53** distinct clusters among 55 proteases, the
largest containing 3. The proteases are not restating one another, so the hazard sum
is not a few enzymes counted many times.

#### `0_02` — geometry before anything else

A matrix ranks bonds, but only a geometry says which bonds are candidates. The
manifest carries identifiers and no geometry column, so this notebook resolves the
geometry of every entry from the EC number and reports what cannot be resolved.

**Findings and decisions.** All panel entries resolve, with `geometry_source`
recording whether the resolution came from the EC number or a name-based fallback.
The notebook also measures the consequence of having ignored geometry: bonds that
are scored today but are inadmissible for that peptidase's geometry are counted
explicitly (`scored_bonds_today`, `forbidden_bonds_today` in `0_09`), which is what
makes the size of the omission visible rather than asserted.

#### `0_03` — BRENDA, and why the label is so much scarcer than the entries

BRENDA looks abundant and is not. Three counts are reported separately because only
the last is a usable label:

1. quantitative entries per protease and parameter type;
2. entries whose **substrate sequence** can be recovered from the free-text
   description;
3. entries whose **scissile bond** can be placed, so that a $P_4 \ldots P_4'$ window
   exists.

Each step loses most of the previous one. An entry reading "hydrolysis of azocasein"
is quantitative and has no sequence; an entry with a sequence but an unstated
cleavage site has no window.

**Findings and decisions.** Of 52 EC rows only **12** have any window-resolved
$k_{cat}/K_M$ at all. The verdicts are:

| BRENDA verdict | EC numbers |
| --- | --- |
| `no_usable_label` | 18 |
| `not_queryable_no_ec` | 12 |
| `scale_anchor_only` | 11 |
| `single_anchor` | 10 |
| `protease_specific_model_possible` | 1 |

`scale_anchor_only` means the entries fix the order of magnitude of the rate but
cannot support a within-protease regression; `single_anchor` means one measurement,
which fixes a point and constrains nothing. Exactly **one** protease in the serum
panel has enough BRENDA kinetics to fit and hold out. That single number is the
reason the analysis had to be widened beyond the serum scope at all.

#### `2_08` — reaction conditions, and whether they can be normalised away

BRENDA values were measured in different laboratories under different conditions,
and temperature changes an enzymatic rate by factors comparable to the effect the
model is trying to detect. The audit is placed in the `2_` series rather than the
`0_` series because it re-runs the regression under each restriction rather than
only counting.

**What is recoverable.** BRENDA records conditions in a free-text comment, not a
field, so they are parsed by pattern: pH for **68.3%** of entries, temperature for
**63.7%**, a mutant marker for **19.3%**, additives for 1.8%. A comment that does not
match contributes a missing value rather than a guess.

**How far apart the conditions are, where it matters.** The panel-wide distribution
is not the relevant quantity, because every correlation is computed *within* a
protease. Within a protease the median temperature span is **12 K** and 100 of 193
codes span at least that; pH spans at least 1.5 units for 30 codes, with an acidic
tail belonging to the cathepsins.

**What a thermodynamic correction would be worth.** Temperature acts through the
activation barrier and has a form, so the correction is computable:

$$\frac{k(T_2)}{k(T_1)} = \exp\!\left[\frac{E_a}{R}\left(\frac{1}{T_1}-\frac{1}{T_2}\right)\right].$$

For $k_{cat}/K_M$ the barrier is that of the transition state relative to free
enzyme plus free substrate, so $E_a$ is an apparent activation energy of a composite
quantity and is not in the data. At $E_a = 50$ kJ/mol, which is $Q_{10} = 1.83$, the
median within-protease span implies a rate factor of **2.06**.

**The measurement that decides it.** Where two laboratories report the same
parameter, for the same enzyme, on the same substrate, **at the same temperature**,
the median fold difference is **5.2×** and the upper quartile **31.5×**. Replicate
groups that differ in temperature scatter by a median of 0.76 in $\log_{10}$ against
0.72 for those that do not — indistinguishable. A temperature correction would
remove a 2.06× term sitting inside a 5.2× noise floor, using a constant the data do
not contain.

**pH is a different case and its outliers are not errors.** The activity profile is
bell-shaped with enzyme-specific ionisation constants and cannot be inverted from
one measurement. A cathepsin assayed at pH 4.5 was measured at its physiological
optimum, the lysosome: that is correct kinetics for a compartment that is not serum.
The response is to label the compartment, not to rescale the value.

**Findings and decisions.** No restriction raises the pooled within-protease
correlation above its unrestricted 0.328 — wild type only gives 0.321, homogeneous
temperature 0.327, physiological pH 0.235 and 37 °C only 0.193, the last two at the
cost of two thirds of the panel. The paired comparison, which holds the protease set
fixed and therefore cannot be blamed on losing the good proteases, agrees: 37 °C
moves 26 shared proteases from 0.242 to 0.212 and physiological pH moves 21 from
0.281 to 0.261. Condition spread also fails to predict which proteases failed
held-out validation, on the 7 proteases carrying both.

Conditions are therefore a real source of variance and not the limiting one. The
limiting one is the between-laboratory scatter that survives at fixed temperature,
which no BRENDA metadata can remove. The decision taken was to **exclude nothing on
condition grounds** — since no exclusion helps and every exclusion costs panel —
while recording the mutant flag and the compartment so that a later analysis can use
them, and to treat single-laboratory datasets such as qPISA as the only route that
would change the answer.

#### `0_04` and `0_05` — the independent sources, compared on one template

These two notebooks are generated from a **single template**, so that datasets
measured by different methods can be read off the same axes. Each answers the same
five questions in the same order: what the dataset physically is; where it
intersects MEROPS; whether the two agree subsite by subsite; how the two sources
differ, stated rather than implied; and how much it enlarges the usable dataset.
`0_11` then places the two side by side.

The reduction that makes this possible is in
`helpers/proteolysis/independent_sources.py`: a set of cleavage windows is tallied
into subsite counts and smoothed with the **same Jeffreys prior** as the MEROPS
matrices, so the comparison is between two estimates of one quantity rather than
between two formats. Subsites an assay cannot resolve are reported as unobserved
rather than scored, because comparing an unobserved subsite compares the prior with
itself and returns perfect agreement from no data.

##### `0_04` — DPP4 qPISA

**The problem it addresses.** MEROPS records **29** cleavages for DPP4 — roughly
one observation per five cells of an 8 × 20 matrix — so the matrix shape is
dominated by the prior and by whichever substrates happened to be studied.

**How the experiment is done.** A proteome digest is incubated with human DPP4 and
compared against a matched no-enzyme control; the readout is the log2 change in
abundance of each peptide. **53 499** peptides are quantified; the most depleted
decile is taken as the cleaved set, with the least depleted decile carried as the
contrast. The substrate sequence is explicit for every observation and the cleavage
site needs no inference: DPP4 is a dipeptidyl-peptidase whose event is by definition
the removal of the N-terminal dipeptide.

**What it costs.** That geometry places the cut after residue 2, so nothing lies
N-terminal of $P_2$: **$P_4$ and $P_3$ do not exist** and only six of eight subsites
can be compared.

##### `0_05` — MMP PICS

**The problem it addresses.** The metallopeptidase matrices rest on 83 to 3 417
cleavages assembled from studies that chose their own substrates.

**How the experiment is done.** Proteome-derived peptide libraries, made by
digesting a proteome with trypsin or with GluC, are exposed to each of nine
metallopeptidases; the newly created N-termini are captured and sequenced. **4 026**
cleavage sites are reported, of which 3 185 carry a complete window. Two libraries
are used rather than one, and kept separate, so that the library composition stays
visible.

**What it costs.** An occurrence records that a bond *was* cut. Whether it was cut
*often* depends on how often that window occurred in the library, and the library
composition is not part of the deposit.

##### What the two sources contribute

| | qPISA | PICS |
| --- | --- | --- |
| proteases | 1 | 9 |
| distinct windows | 5 095 | 3 636 |
| median windows per MEROPS cleavage | **175.7** | 1.6 |
| subsites resolved | 6 of 8 | 8 of 8 |
| measurement type | depletion against control | occurrence count |

##### Findings and decisions

**Agreement with MEROPS tracks the measurement type, not the amount of evidence.**
This is the result of `0_11` and it is the most consequential thing the source
audits produced.

| Source | median rank agreement | median divergence (bits) | top residue agrees |
| --- | --- | --- | --- |
| PICS | 0.62 – 0.81 per subsite | 0.05 – 0.09 | 36 of 72 (50%) |
| qPISA | 0.04 – 0.55 per subsite | 0.10 – 0.25 | 1 of 6 (17%) |

PICS contributes 1.6 windows per MEROPS cleavage and agrees closely. qPISA
contributes **176** per cleavage — two orders of magnitude more — and agrees much
less, with $P_3'$ at 0.04 and $P_4'$ at 0.11. Plotting agreement against evidence
ratio across all ten proteases gives no positive trend.

**Neither figure is a single number, and the distribution behind each is shown.**
A source profiling nine proteases yields nine comparisons per subsite and one
profiling a single enzyme yields one, so both notebooks report three layers
instead: a bootstrap over the source's own windows, the source's own strata as
points, and the pooled value as a dash. For qPISA the band is narrow — rank
agreement at $P_1'$ between 0.51 and 0.59 across resamples — so its low agreement
is a property of the data and not of the sample size, and at $P_3'$ the band
straddles zero. Its strata run from the least to the most depleted fifth of the
cleaved set and show no rising trend, which rules out the explanation that MEROPS
records only the strongly cleaved sites. For PICS the two peptide libraries differ
by 0.05 to 0.12 per subsite, inside the band, so library composition is not driving
its agreement either.

The explanation is what each source measures. PICS and MEROPS both count
**occurrences of a cleavage**; qPISA measures **how much of a peptide disappeared**,
which is nearer to a rate. A MEROPS matrix therefore reproduces well what another
occurrence-counting experiment finds and reproduces poorly what a
depletion-weighted experiment finds.

**That is a statement about the project, not about these two datasets.** The window
score is built from occurrence counts and the calibration wants to predict a rate.
At the one place where the two kinds of measurement can be compared directly they
disagree, which is consistent with the `2_` series reaching a pooled within-protease
correlation of 0.36 with only two proteases surviving held-out validation, and it
locates the reason in the evidence rather than in the fitting.

**Practical consequence.** An independent source is worth acquiring for the *kind*
of measurement it makes, not for its size. Another proteome-scale occurrence dataset
would mostly restate MEROPS; a dataset weighting cleavages by how much substrate was
consumed would not.

#### `0_06` — serum half-lives, and why plasma is kept out of serum

This is the label for validation B, derived as $k_{obs} = \ln 2 / t_{1/2}$.

**How the sources differ.** PEPlife2 is a curated collection of half-lives
transcribed from papers; DRAMP records stability as free text that has to be parsed.
More importantly, serum and plasma are **different matrices**: plasma retains the
coagulation cascade, serum has consumed it, so the protease population differs in
exactly the enzymes this model is about.

**Findings and decisions.**

| Source | Matrix | Observations | Sequences | Publications | Verdict |
| --- | --- | --- | --- | --- | --- |
| PEPlife2 | human serum | 32 | 19 | 16 | `rank_test_only` |
| PEPlife2 | human plasma | 378 | 145 | 104 | `kept_separate_from_serum` |
| DRAMP | serum mentioned | 64 | 49 | 11 | `rank_test_only` |

Nineteen human-serum sequences drawn from sixteen different publications cannot
calibrate anything: the between-publication variance in assay conditions is not
separable from the between-peptide variance that the model is supposed to explain.
The verdict `rank_test_only` is recorded here, in the audit, and `3_01` obeys it —
which is why `3_01` reports a rank correlation as its validation and treats the
fitted offset as a unit conversion rather than as a result.

The plasma set is an order of magnitude larger and is deliberately **not** merged.
Merging would buy statistical power by changing the biological question.

#### `0_07` — abundance is a prior, not a concentration

**How the two methods differ.** Mass spectrometry measures peptides and
under-reports proteins that ionise poorly or sit in complexes; immunoassays depend
on an antibody and can over-report. They are not two estimates of the same number
with independent noise; they are two different measurements.

**Findings and decisions.** Of 54 panel proteases, **41** have a mass-spectrometry
value, **27** an immunoassay value, **42** at least one and **26** both. On the
proteins measured by both, the median absolute $\log_{10}$ ratio between the methods
is **0.485** — a factor of about **3.1** on the median protein, with a long tail.
Any weighting built on these numbers inherits that uncertainty.

The deeper point is recorded as `usable_as_active_concentration = False`. A
circulating protease may be a zymogen, may be bound to an inhibitor such as
antithrombin or α2-macroglobulin, or may be free, and an abundance measurement
counts all three states identically. The relation between abundance and active
concentration is not a scaling factor, so the value enters the model as an
**abundance prior** $w_\pi$ and never as $c_\pi$. Verdict: `prior_only_not_a_label`.

#### `0_08` — the control panel as an independent criterion

Audits `0_01` to `0_07` select proteases by data availability, which is a criterion
about the literature and not about the experiment. This notebook applies the
independent one: which proteases the project's own control panel actually names.

**Findings and decisions.** The panel's enzymes are largely **bacterial**, and their
matrices come from the pathogen datasets: OmpT `A26.001` from *E. coli* ATCC 11775
(54 cleavages, rank stability **0.869**), LasB `M04.005` from *P. aeruginosa* PA01
(70 cleavages, stability **0.890**). Both are markedly more stable than the human
serum panel's median of 0.726, because their literature is concentrated in a few
systematic studies rather than spread across incidental observations.

This produces the split that runs through the rest of the analysis: the bacterial
enzymes are where the *published bonds* are, and the human enzymes are where the
*model* is. Keeping the two lookups separate — `any_matrix_for` for validating the
scoring rule, the restricted lookup for the serum model — is what allows the
bacterial evidence to be used without contaminating the panel.

Thirteen panel proteases have no EC number or an ambiguous UniProt mapping and are
recorded here as **blockers with a reason**, not as zero counts.

#### `0_09` — the decision table, and which proteases are rejected

The synthesis joins the seven audits and applies the decision rule on the three
axes. It is the notebook that says out loud which proteases cannot be carried
forward and why.

**Findings and decisions.**

| Axis | Status | Proteases |
| --- | --- | --- |
| mapping | `mapped_reviewed` | 51 |
| | `unresolved_multiple_human_mappings` | 3 |
| | `unresolved_no_human_mapping` | 1 |
| specificity | `merops_matrix_only_weak` | 34 |
| | `independent_dataset_can_replace_merops` | 9 |
| | `merops_matrix_only_usable` | 7 |
| | `no_usable_specificity` | 5 |
| kinetics | `no_kinetics` | 34 |
| | `single_measurement` | 13 |
| | `absolute_anchors` | 7 |
| | `relative_dataset` | 1 |

Read together: **34 of 55** proteases have no kinetic label of any kind, and a
further 13 have exactly one measurement. Five have no usable specificity at all. The
nine that an independent dataset can replace are the PICS metallopeptidases plus
DPP4 — no other panel entry has a window-resolved dataset independent of MEROPS.

The notebook then states what each validation level can actually be attempted:

| Level | Question | Status |
| --- | --- | --- |
| A, MEROPS → kinetics | does the window score predict catalytic efficiency? | testable for **1** protease, on relative labels; 46 observations panel-wide are on an absolute scale, not enough to calibrate one protease |
| A, specificity only | can an independent dataset replace or check a matrix? | replaceable for **9** proteases |
| B, kinetics → serum stability | do susceptible peptides degrade faster? | rank test only |

This table is the reason the analysis did not stop at the serum panel. Validation A
was testable for one protease; that is not a result, it is an absence of one.

#### `0_10` — what widening the scope actually adds

The serum scope came from the calibration specification, which is about serum. The
hypothesis the `2_` notebooks test is different: whether a MEROPS matrix predicts
the kinetics of **its own** enzyme. That is a question about matrices, and it has no
reason to respect a serum boundary.

**The cross-check.** The wide mapping runs EC → UniProt → MEROPS and the manifest
ran MEROPS → UniProt → EC, so agreement between them is evidence and not tautology.
`0_10` confirms that the two agree on all **42** shared assignments and that the wide
extraction reproduces the narrow one row for row.

**Findings and decisions.** The expansion reaches **101** MEROPS codes. The codes it
adds are not marginal: cathepsin L gains **67** measured substrates, cathepsin D
**59**, cathepsin V 11, cathepsin B 8 — every one of them at **zero** in the serum
scope, while their MEROPS matrices are among the largest in the database
(cathepsin S at 3 126 cleavages, cathepsin L at 2 956, cathepsin E at 1 596). These
are well-characterised enzymes that the serum filter had excluded for a reason that
has nothing to do with how well they are known.

The cost is stated in *The two parallel series*: sevenfold more evidence for about a
fifth of the pooled correlation. The decision was to carry the expanded scope and
keep both series, so the cost stays visible rather than being absorbed.

### Exopeptidases as a separate process

`1_02` treats exopeptidase action as what it is — a **ladder** of successive
single-residue or dipeptide removals from one terminus — rather than as an
endopeptidase with an unusual preference. An exopeptidase cannot cut internally, so
giving it the endopeptidase bond set would let it produce fragments the enzyme
cannot make.

**How it is measured.** For each of 10 exopeptidases in the panel the notebook
descends the ladder rung by rung, scoring the event at each step, and compares the
cost of trimming against the cost of one internal cut **at matched residues
removed**. Comparing at the end of the ladder is uninformative — every enzyme
converges there — which is why the matched-removal comparison exists.

**Findings and decisions.** The ladder runs a mean of **10.7** rungs before the
score is exhausted, and the score falls by **0.81** nats from first rung to last, so
processivity is not free: each successive removal is less favoured than the last.
Against an internal cut at matched residues removed, trimming costs a median
**0.438** $\log_2$ MIC against **0.359** for the internal cut, and trimming is the
cheaper option in a mean **37%** of cases.

Exopeptidases are therefore **competitive with internal cleavage, not background**,
and must be modelled as their own process. The external check is qPISA: the MEROPS
matrix of DPP4 encodes the known Pro/Ala preference at $P_1$ with a score separation
of **2.095** nats, against **2.56** $\log_2$ measured by qPISA — the right motif, at
roughly the right magnitude, from a matrix built on 29 cleavages against a dataset
of 53 499 observations.

### Definition of the metrics

Everything in this subsection is implemented in
`pep_compass.optimization.components.helpers.proteolysis` and
`...helpers.activity`, and is covered by the tests listed at the end of this
document. The definitions are given in the form the code uses, not in a more
general form that would not match it.

#### The window score

##### Subsites, masks and the scoring formula

A cleavage event is located at a **bond** $b$, meaning the peptide bond between
residues $b$ and $b+1$ in 0-based indexing. The eight subsites are read off the
sequence around that bond: $P_4 \ldots P_1$ are residues $b-3 \ldots b$ and
$P_1' \ldots P_4'$ are residues $b+1 \ldots b+4$.

Near a terminus some subsites fall outside the peptide. They are **masked out**
rather than padded, because padding with any residue asserts an observation that
was not made. With $m_p \in \{0,1\}$ the observability of subsite $p$ and $a_p$ the
residue occupying it,

$$S^{M}_{\pi,b} \;=\; \sum_{p\,:\,m_p=1} \Big( M_{\pi,p,a_p} \;-\; \log q_{a_p} \Big),$$

where $M_{\pi,p,a}$ is the smoothed log-probability of residue $a$ at subsite $p$
for protease $\pi$, and $q_a$ is the background frequency of §*The reference
distribution*. The score is in **nats of log-odds**.

Two properties follow directly and matter for reading any figure:

- The score is a **sum over observable subsites**, so a bond near a terminus with
  four observable subsites is not on the same scale as a central bond with eight.
  Wherever bonds of different observability are compared, the number of observed
  subsites is carried alongside and used as a control.
- A **negative** score means the residues at that bond are less likely under the
  protease's profile than under the human proteome background: the bond is
  actively disfavoured. It does not mean "no data".

##### The two substrate conventions: reporter and quenched

Almost every kinetic measurement in BRENDA was made on a **synthetic** substrate,
and synthetic substrates come in two designs that differ in exactly the property
this analysis needs. The distinction runs through the whole series, so it is worth
setting out concretely.

**A reporter substrate** carries a chromogenic or fluorogenic leaving group at the C
terminus:

```
Suc - Ala - Ala - Pro - Phe - pNA
                          ^     ^
                         P1   the reporter
```

The enzyme cuts the bond between the last peptide residue and the reporter, and the
signal is the released reporter. Consequences:

* the **bond is known by construction** — it is the C-terminal bond, nothing has to
  be inferred;
* **P1 is the last peptide residue**, and the non-prime side reaches back at most
  four residues;
* the **prime side is chemistry, not peptide**, so $P_1' \ldots P_4'$ carry no
  residue and contribute nothing to the score.

`resolvable_subsites` returns $\min(\text{length}, 4)$ for such a substrate: at best
half a window.

**A quenched substrate**, also called internally quenched or FRET, carries a
fluorescence donor at one end and a quencher at the other with a real peptide
between them:

```
Abz - Ala - Ala - Lys - Phe - Phe - Ser - Arg - Gln - EDDnp
      donor                                            quencher
```

Cleavage anywhere inside separates donor from quencher and the signal rises.
Consequences:

* residues exist on **both sides** of whatever bond was cut, so all eight subsites
  can be occupied;
* but the database records the rate and **not which bond was hydrolysed**, so the
  bond is unknown.

The two designs therefore trade the same two properties against each other:

| | bond | window |
| --- | --- | --- |
| reporter | known | at most 4 of 8 subsites, non-prime only |
| quenched | **unknown** | up to 8 of 8, both sides |

This is why `2_05` and `2_06` exist. The reporter substrates alone never consult the
prime half of any matrix — the median reporter populates three of eight subsites —
and the quenched substrates are the only ones that would, if their bond could be
established. Establishing it is the subject of those two notebooks, and the rule
must not be "take the bond the matrix scores highest", because scoring a bond chosen
by the score is circular. The score of a quenched substrate is instead
$S^{\max} = \max_b S^{M}_{\pi,b}$, the best the protease could do with that
substrate, which uses no knowledge of the answer.

##### Reporter substrates and the latent bond

Many BRENDA substrates are quenched fluorogenic reporters, where the measured
signal is the separation of a donor and a quencher and the cleaved bond is often
not stated. Choosing the bond by assumption and then scoring it would make the
correlation between score and rate circular.

Two devices avoid this:

- For reporter substrates whose construct implies the geometry — P1 is the last
  residue of the recognition sequence and the prime side is the leaving group —
  `reporter_window_indices` builds the index array directly, with negative indices
  marking the unobservable prime-side positions, and `masked_score_from_indices`
  scores it. Nothing is invented; the unobserved subsites are simply masked.
- Where the bond is genuinely unknown it is treated as a **latent variable**:
  every admissible bond is scored and the substrate's score is
  $S^{\max} = \max_b S^{M}_{\pi,b}$. The maximum is a statement about the best the
  protease could do with this substrate, which is the quantity a rate should track,
  and it uses no information about the answer.

#### Event geometry

A peptidase does not attack every bond. `EventGeometry` distinguishes
aminopeptidase, dipeptidyl-peptidase, tripeptidyl-peptidase,
peptidyl-dipeptidase, carboxypeptidase and endopeptidase, and
`allowed_cut_positions(length, geometry)` returns the admissible bonds for each.

Geometry is resolved from the EC number, primarily by sub-subclass:

| EC sub-subclass | Geometry |
| --- | --- |
| `3.4.11` | aminopeptidase |
| `3.4.14` | dipeptidyl- or tripeptidyl-peptidase |
| `3.4.15` | peptidyl-dipeptidase |
| `3.4.16`–`3.4.18` | carboxypeptidase |
| `3.4.21`–`3.4.25` | endopeptidase |

Sub-subclass `3.4.14` mixes two geometries, so exact-EC overrides take precedence
over the sub-subclass rule: `EC_NUMBER_GEOMETRY` maps `3.4.14.9` and `3.4.14.10` to
tripeptidyl. The override exists because TPP1 (`S53.003`) was typed as dipeptidyl
by the sub-subclass rule alone, which is wrong and changes which bonds are
admissible.

> **Findings and decisions.** All **57** panel proteases resolve to a geometry;
> `geometry_source` records whether the resolution came from the EC number or from a
> name-based fallback, so a reader can discount the fallbacks.

#### The cleavage intensity and the Poisson process

##### Intensity

Each (protease, bond) pair is an independent exponential clock with intensity

$$\lambda_{\pi,b} \;=\; w_\pi \, \exp\!\big(\alpha + \beta\, S^{M}_{\pi,b}\big),$$

where $w_\pi$ is the abundance weight, $\beta$ converts the score to a log-rate,
and $\alpha$ is a global offset.

The roles of the two constants are asymmetric and this is worth stating plainly.
$\alpha$ is a **common factor** of every intensity in the panel: it sets the unit
of time and cancels completely from every bond probability, every ordering and
every ratio. $\beta$ does **not** cancel: it decides how strongly the score
reorders events, and at $\beta = 0$ the model degenerates to a hazard proportional
to the number of admissible bonds. That degenerate setting is used throughout as
the **internal control**, under the name `concentration_only`.

$\beta = 0.404 \ln 10$ per nat. The figure $0.404$ per decade is the fitted slope of
kallikrein 1 (`S01.160`), the one per-protease estimate that survived held-out
validation; it is converted from $\log_{10}$ to nats and then applied to the whole
panel. That extrapolation is an assumption, stated here and repeated wherever the
slope is used.

$\alpha$ is fitted in `3_01` against measured serum half-lives: $-12.269$ for the
uniform weighting and $-17.642$ for the concentration weighting.

##### Three weighting variants, carried everywhere

| Variant | $w_\pi$ | $\beta$ | What it isolates |
| --- | --- | --- | --- |
| `uniform` | 1 | $0.404\ln 10$ | specificity alone, all proteases equally present |
| `concentration` | $a_\pi/\mathrm{median}(a)$ | $0.404\ln 10$ | specificity and abundance together |
| `concentration_only` | $a_\pi/\mathrm{median}(a)$ | $0$ | abundance and bond count alone — the control |

The third is not a model anyone would propose. It exists so that any claimed effect
can be checked against the version of the model in which the specificity matrices
carry no information at all. Where `concentration_only` reproduces an effect, the
matrices did not cause it.

##### The process

Cleavage is modelled as a continuous-time Markov jump process over the set of
fragments, simulated by the Gillespie construction: the waiting time to the next
event is exponential with rate $\Lambda = \sum \lambda$, and the event that fires is
drawn with probability $\lambda_i / \Lambda$. Event times are **exact** — no time
step is chosen, so none can be too coarse, which was the explicit requirement.

Because cleavage is first order in the peptide, molecules are independent and a
population is the distribution over repeated single-molecule trajectories. The
first event needs no simulation at all: `first_cut_distribution` returns the exact
bond probabilities and the expected waiting time $1/\Lambda$ in closed form, and
that exact result is what `4_01` uses.

#### Potency of a fragment mixture

The activity of a digest is summarised by the potency sum

$$\psi \;=\; \sum_i \frac{C_i}{\mathrm{MIC}_i},$$

the fractional inhibitory concentration of the mixture, which is additive over
fragments and equals 1 when the mixture is exactly inhibitory.

Two conventions are required to make it computable, and both were wrong in the
first version of the notebooks.

##### The dose convention

Each fragment inherits the molar concentration of the parent, since one molecule
of parent yields one molecule of each product. The **dose** is expressed in
multiples of the parent's own MIC, so the intact peptide sits at $\psi = 1$ by
construction. Dosing at unit molar concentration instead — the original choice —
puts every peptide near $\psi \approx 0.02$, two orders of magnitude below the
threshold at which the thresholded transforms operate, which silences `threshold`
entirely and pushes `hill_kill_rate` and `soft_threshold` into their linear tails
where they become indistinguishable from `potency_sum`.

##### The minimum-length guard

A fragment shorter than a guard length is treated as **inactive** rather than given
a MIC prediction the model was not trained to make. Without the guard the additive
rule produces the wrong sign on a digest whose outcome is published (§6).

The guard is calibrated in `4_03` against two digests of LL-37 with published
activity outcomes, and the admissible range is 9 to 21 residues. It is a
biologically calibrated bound on the MIC model's domain, not a fitted parameter,
and §6 records that it does not transfer to short peptides.

#### Activity transforms

`activity_transform_registry` is a registry in the sense of
`pep_compass.registry.Registry`, so a new metric is added by decorating a factory
and is then available by name to every notebook. Six are registered:

| Name | $A(\psi)$ | Parameters | What it assumes |
| --- | --- | --- | --- |
| `potency_sum` | $\psi$ | — | activity is linear in the inhibitory fraction |
| `log_potency` | $\log \psi$ (floored at $10^{-6}$) | `floor` | activity is linear in log concentration; **signed** |
| `threshold` | $\mathbf{1}[\psi \ge 1]$ | `inhibitory_potency` $=1$ | activity is all-or-nothing at the MIC |
| `soft_threshold` | sigmoid in $\log \psi$ | steepness, midpoint | as above, with a finite transition |
| `hill_kill_rate` | $E_{\max}\psi^{H}/(1+\psi^{H})$ | $H=1$, $E_{\max}=1$ | pharmacodynamic kill rate saturating at $E_{\max}$ |
| `net_kill_rate` | $\mathrm{hill} - \mu$ | $\mu = E_{\max}/2$ | as above, minus bacterial growth; **signed** |

`net_kill_rate` places its zero at $\psi = 1$ by choosing $\mu = E_{\max}/2$, which
is exactly the definition of the MIC: below it the population grows, above it the
population dies. That is the interpretation the analysis was asked for — a smaller
MIC means faster death, not merely a better number.

The activity budget over a horizon is
$B(T) = \int_0^T A(t)\,\mathrm{d}t$, evaluated on a fixed time grid with the
trapezoid rule.

> **Findings and decisions.** Because two transforms are **signed**, a budget ratio
> $B/B_{\text{intact}}$ inverts their direction: a negative budget divided by a
> negative reference is largest for the peptide that lost the most activity. The
> comparison is therefore made on the mean activity $B(T)/T$ against the intact
> peptide's activity, never on a ratio.

### Construction of the tests — summary

This subsection is the short account: what is being tested, on what, against what
null. The subsection that follows gives the same tests in the detail required to
reimplement them, and the two are meant to be read in that order.

**Held-out validation of the score-to-rate relation (`2_01`, `2_03`, `2_04`,
`2_06`).** For one protease, take its measured substrates, score every substrate's
window, and ask whether the score ranks the measured catalytic efficiencies.
Because substrates within a protease's literature are frequently minor variants of
one another, the substrates are first clustered by sequence similarity and the
train/test split is made **between clusters**, so a near-duplicate of a training
substrate cannot appear in the test set. The split is repeated 100 times. The
verdict uses the median held-out rank correlation, the share of splits on which the
correlation was positive, and a permutation null constructed for that median.
Multiplicity across proteases is controlled by Benjamini–Hochberg.

**Recovery of published cleavage sites (`2_02`, `2_05`, `3_03`).** For one peptide
and one protease, score every admissible bond and ask whether the bonds reported
cleaved in the literature outscore the others. The statistic is the probability
that a published bond outscores an arbitrary unpublished one, which is an AUC. The
null redraws, for each row, as many bonds as it has published sites, uniformly from
the admissible bonds. A second control replaces the score with the number of
observable subsites, to establish that a high AUC is not simply central bonds
scoring higher.

**Calibration and validation of the time axis (`3_01`).** Fit the single global
offset $\alpha$ so that modelled hazards match measured serum half-lives, then ask
separately whether the modelled hazard **orders** those half-lives. Fitting an
offset always succeeds and proves nothing; the ordering test is the one that can
fail. Peptide length is carried as a competing predictor.

**Discrimination of annotated peptides (`3_02`).** Ask whether peptides a source
calls serum-stable receive a different hazard from those called labile, across
unrelated scaffolds, and separately whether a parent and its engineered variant
move in the published direction. The second comparison is paired and two of the
pairs are length-matched to the residue, so a difference cannot be a length
artefact.

**Cleavage-product activity (`4_03`).** Ask whether the MIC predictor ranks
cleavage products that the literature reports as retaining activity above products
from a digest reported to abolish it, and whether the additive mixture rule
reproduces the digest-level outcome. Both comparisons are repeated with peptide
length held fixed.

**Outcome distributions (`4_01`, `4_02`).** These are not hypothesis tests. `4_01`
computes the first-cut distribution exactly and `4_02` simulates the full chain many
times per peptide, so that the outcome of cleavage is reported as a distribution
over cleavage patterns rather than as an expectation.

### Construction of the tests — full detail

Everything below states the estimator, the null, the resampling unit and the
guard conditions, in the form the code implements. Where the implementation departs
from the ideal procedure, the departure is named here rather than in §6.

#### Clustering of substrates, and why the split is between clusters

A protease's measured substrates are not independent draws. A single paper
frequently reports a positional scan — one scaffold with one residue varied — and
BRENDA records each variant as its own entry. A random train/test split over rows
would place a variant in the test set whose near-twin is in the training set, and
the held-out correlation would then measure memorisation.

Substrates are therefore grouped before splitting. The similarity between two
specificity profiles is their correlation; `cluster_profiles(similarity, threshold)`
performs **average-linkage agglomeration on the correlation distance $1-r$**, cut so
that members of a cluster have an average correlation of at least the threshold,
which defaults to **0.8**. The cluster, not the substrate, is the resampling unit
everywhere downstream.

#### Repeated held-out evaluation

For one protease with at least 3 measured substrates:

1. **Seed.** The random generator for this protease is seeded with
   `SEED + zlib.crc32(merops_code)`. This is not decoration: an earlier version
   drew from one shared generator, so a protease's splits depended on how many
   proteases had been processed before it, and the same 67 substrates gave a
   held-out $\rho$ of 0.532 or 0.800 depending on processing order. Seeding per code
   makes the result a property of the protease.
2. **Split, 100 times.** Clusters are assigned to train or test. The **largest
   cluster is reserved for training**, and a cluster whose addition would push the
   test set past its size limit is skipped. Both halves must hold at least **6**
   substrates or the split is discarded. Without the reservation rule the splitter
   pushed up to 75% of substrates into the test set — for `S01.160`, 17 train
   against 50 test — which is not a held-out evaluation.
3. **Statistic.** On each valid split, the Spearman correlation between the window
   score and the measured $\log k_{cat}/K_M$, computed **within** the protease and on
   the test clusters only. The protease's statistic is the **median** over splits.
4. **Stability.** The share of splits on which the correlation was positive is
   reported beside the median. A median of 0.6 with 55% positive splits is a
   different object from a median of 0.6 with 100% positive splits, and the verdict
   uses both.
5. **Null.** The labels are permuted and pushed through the **same stored
   partitions**, so the null median is built from exactly the split structure that
   produced the observed median. Permuting before splitting would also destroy the
   cluster structure and give an optimistically narrow null.
6. **Multiplicity.** Benjamini–Hochberg across the proteases tested, reported as
   `q_value`.

#### Recovery of published cleavage sites

For one (peptide, protease) row with verified published bonds:

1. **Candidate set.** Every bond the geometry admits and for which a finite score
   exists. Terminal bonds excluded by the geometry are not candidates and are not
   counted in the denominator.
2. **Estimator.** With $\mathcal{P}$ the published bonds and $\mathcal{N}$ the rest,

   $$\mathrm{AUC} \;=\; \frac{1}{|\mathcal{P}||\mathcal{N}|}\sum_{p\in\mathcal{P}}\sum_{n\in\mathcal{N}}\Big(\mathbf{1}[s_p > s_n] + \tfrac{1}{2}\mathbf{1}[s_p = s_n]\Big).$$

   **Ties count as half a win.** This is not a detail: the switched-off control
   produces exactly tied intensities, and a strict inequality scores that case 0
   rather than 0.5, reporting a flat, uninformative profile as an inverted one.
3. **Pooled statistic.** The mean AUC over rows.
4. **Null.** For each row, redraw $|\mathcal{P}|$ bonds uniformly without
   replacement from that row's candidates, recompute the pooled mean, repeat
   $B = 2000$ times. The null is conditional on the same rows and the same candidate
   sets, so it controls the bond labelling.
5. **Position control.** Replace the score by the number of observable subsites and
   run the identical statistic. Published sites in short peptides are not at the
   termini and neither are high scores, so this control establishes whether the
   result is available without any specificity. It is not a null; it is a competing
   explanation.
6. **Evidence grading.** Rows whose bonds were reconstructed by homology are
   computed and displayed but marked, and the conclusion rests on the rows the
   sources publish as bonds.

##### The rank of a published bond

Beside the AUC, the site-recovery notebooks report the **rank** of the published
bond, and it answers a blunter question than the AUC does.

For one peptide and one protease, every bond the geometry admits is scored and the
bonds are sorted from the highest score down. The published bond's rank is its
position in that list:

* **rank 1** means the model's single best guess is exactly the bond the experiment
  found;
* **rank 18 of 18** means the model ranked the true site last.

Where a source publishes several bonds for one peptide, the reported `best_rank` is
the best rank any of them achieved, because a model that nominates one real site out
of three has found a real site.

The two statistics fail differently and are reported together. The AUC uses every
bond and so is stable, but it can sit high while the top-ranked bond is wrong — a
published bond beating 90% of the alternatives still loses to the other 10%. The
rank ignores everything except the head of the list, which is the part `4_01` acts
on when it nominates a first cleavage. For oncocin the two agree: AUC 1.00 and rank
1 of 18 under the concentration weighting, meaning the published site beat every
alternative and was also the model's first choice.

#### Two-tier site recovery

The same estimator is applied at two levels, and they answer different questions.

- **Single enzyme.** One protease, its own matrix, from the dataset where it has
  the most cleavages. This tests the scoring function and legitimately uses
  bacterial enzymes, because the published experiments used them.
- **Panel.** The assembled `ProteaseKinetics` object that `4_01` uses, ranked by
  the first-cut probability per bond rather than by a raw score. This tests the
  deployed model. For a whole-serum measurement the human serum panel is the
  correct model, and `concentration_only` is run beside it as the control.

#### Calibration of the time axis, and its separate validation

These are two procedures and conflating them would be the easiest way to
manufacture a false positive.

- **Calibration.** One global $\alpha$ per weighting is fitted by least squares so
  that $\log(1/\Lambda)$ matches measured $\log$ half-lives over the 68 sequences
  with both. A single offset against 68 points will always fit something; the
  residual scatter is reported as `residual_sd_log10` and the fitted range as a
  confidence interval.
- **Validation.** Separately, the Spearman correlation between the modelled hazard
  and the measured degradation rate, with a permutation null. This is the test that
  can fail, and it is reported with its own p-value computed as
  $(1+k)/(1+B)$.
- **Competing predictor.** Peptide length is correlated with the same measurements,
  because if length alone predicts better than the model then the model has added
  nothing.

#### Discrimination, unpaired and paired

- **Unpaired.** One-sided Mann–Whitney $U$ for the labile group exceeding the
  stable group, reported as $\mathrm{AUC} = U/(n_1 n_2)$. With 4 stable and 6 labile
  peptides the most extreme attainable result is $p = 1/\binom{10}{4} = 0.0048$, and
  that bound is stated **before** the result, so a null result is not read as a
  failure of power that better luck would have fixed.
- **Paired.** Fold change in modelled time from parent to variant, per weighting.
  With four pairs a perfect sign test gives $p = 1/2^4 = 0.0625$, so the effect size
  carries the interpretation rather than the p-value. Two of the pairs are
  length-matched to the residue, and `concentration_only` must return exactly
  $1.00$ on those by construction — which it does, establishing that any other
  variant's departure from 1 comes from the matrices.

#### Cleavage-product activity, and the length confound

- **Raw comparison.** One-sided Mann–Whitney on predicted MIC, products reported
  as retaining activity against products of a digest reported to abolish it.
- **Length control.** Every contiguous window of the parent peptide down to four
  residues is scored, and each product is placed as a **percentile among the
  windows of its own length**. This holds length fixed by construction. The
  retained products are long and the abolished ones short, so without this control
  a predictor that read nothing but length would separate them perfectly.
- **Mixture rule.** The potency ratio of a digest to its parent is recomputed
  across a sweep of the minimum-length guard, and the guards that reproduce **both**
  published digests at once are reported as a range rather than a point.
- **Strain sweep.** Where the published result was measured on a species other than
  the reference strain, the comparison is repeated on all 34 APEX strains, so the
  obvious escape — "wrong strain" — is tested rather than argued.

#### Outcome distributions

- **Exact first cut.** `first_cut_distribution` gives the bond probabilities and the
  expected waiting time in closed form. No simulation error enters `4_01`.
- **Full chain.** `4_02` draws 60 trajectories per peptide per weighting and keeps
  each as its own row. The probe is stratified into potent, middle and weak tiers by
  predicted parent MIC, 12 peptides each, so the extremes are present by
  construction rather than by luck.
- **Spread.** Reported as the interdecile range and the share of trajectories
  ending at exactly zero potency. A ratio of extremes is undefined here, because a
  trajectory whose fragments all fall below the guard has potency exactly zero.
- **Transform agreement.** Spearman rank correlation between transforms on the mean
  activity, over peptides.

#### The permutation p-value estimator

A permutation p-value must be computed as

$$p \;=\; \frac{1 + \#\{b : T_b \ge T_{\mathrm{obs}}\}}{1 + B},$$

because the observed statistic is itself one of the attainable arrangements. The
naive proportion can return exactly zero, which is not a probability and misstates
the resolution of the test: with $B = 2000$ the smallest supportable statement is
$p \le 1/2001 \approx 0.0005$.

`3_01` and `3_03` use the corrected estimator. `2_01`, `2_03`, `2_04` and `2_06` do
**not** — they compute the naive proportion — and therefore print `p = 0.000` where
`p ≤ 1/(B+1)` is the correct statement. This is recorded as a known defect in §6
rather than silently corrected in the prose.

### Execution order and artifact dependencies

The notebooks form a chain through `results/`, not through shared memory, so any
stage can be re-run alone provided its inputs exist. There is no precompute step in
this directory; the two artifacts that are produced once are described under
*Human reference proteome* and *BRENDA kinetics*.

| Stage | Reads | Writes | Consumed by |
| --- | --- | --- | --- |
| `0_01`–`0_08` | raw sources, `data/merops/`, manifest | one coverage table per source | `0_09` |
| `0_09` | the coverage tables | `0_14_dataset_definition.csv` | scope decisions |
| `0_10` | both BRENDA extractions | `0_10_expanded_kinetics_integration.csv` | `2_04` onward |
| `2_01`–`2_03` | narrow BRENDA scope | per-protease verdicts, splits | `2_04` |
| `2_04`–`2_06` | expanded scope | per-protease verdicts, splits | `2_07` |
| `2_07` | verdicts, geometry, abundance | **`2_07_model_panel.csv`** | every `3_` and `4_` notebook |
| `3_01` | model panel, half-lives | **`3_01_time_axis_calibration.csv`** ($\alpha$) | `3_02`, `3_03`, `4_01`, `4_02` |
| `3_02`, `3_03` | model panel, $\alpha$, control panel | discrimination and site-recovery tables | §6 |
| `4_01` | model panel, $\alpha$, APEX tables | first-cut landscape | §6 |
| `4_02` | model panel, $\alpha$, APEX, guard from `4_03` | outcome distributions | §6 |
| `4_03` | control panel, APEX ensemble | fragment outcomes, guard range | `4_02` |

`2_07_model_panel.csv` is the single junction of the analysis: 57 proteases, each
with a geometry, an abundance where one exists, a validation verdict and a
validation tier. Everything after it is a consequence of that file.

Note the one backward dependency: `4_02` consumes a guard range calibrated in
`4_03`, so `4_03` must run first although it is numbered later.

#### The two parallel series

`2_02`/`2_03` and `2_05`/`2_06` implement the same procedures on the narrow and the
expanded scope. They are kept in parallel rather than replaced, because widening
the data changes what is being measured and the cost of the widening is itself a
result:

| Quantity | Narrow scope | Expanded scope |
| --- | --- | --- |
| proteases with ≥ 3 substrates | 12 | **57** |
| proteases with ≥ 6 substrates | 9 | 36 |
| pooled observations | 113 | **773** |
| pooled within-protease $\rho$ | **0.455** | 0.356 |
| median subsites populated | 3 | 5 |

The expansion multiplies the evidence by about sevenfold and costs about a fifth of
the pooled correlation. The decision taken was to carry the expanded scope forward,
on the grounds that a weaker relation measured on 57 proteases constrains the model
more than a stronger one measured on 12.

#### Environment

The notebooks read precomputed artifacts and run in seconds to a few minutes, apart
from `4_02` and `4_03`, which call the APEX ensemble and take a few minutes more.
`DEVICE` selects CUDA when it is available. The commands are in
[Execution](#execution).

### What the results license

The chain of §1 did not survive intact, and the pattern of which links held is
itself the main result.

#### Confirmed against measurements

**Link 1 holds: the score ranks bonds within a peptide.** In `3_03` the pooled AUC
over the rows with verified published bonds is **0.913** against a permutation null
centred at 0.499 with a 95th percentile of 0.632, $p = 0.0005$, which is the
smallest value $B = 2000$ can support. The position control returns **0.517**, so
the result is not an artefact of central bonds carrying more observable subsites.
At panel level the human serum model ranks the published dominant cleavage site of
oncocin **first of eighteen** candidate bonds with AUC 1.000 under the
concentration weighting, while the switched-off control ranks it fifth at chance.

**Link 5 holds, with a length caveat.** In `4_03` every LL-37 product the
literature reports as retaining activity is predicted more potent than every
product of the digest reported to abolish activity: AUC 1.00, exact one-sided
$p = 0.014$. With length held fixed the separation falls to AUC 0.81, $p = 0.10$, so
length does part of the work and the length-controlled test cannot reach
significance with eight products. The ordering is right in detail as well: `RK-31`
and `KS-30`, reported in the literature as more active than the parent, are the two
most potent products predicted.

#### Not confirmed

**Link 2 is supported in aggregate and for two proteases only.** The pooled
within-protease correlation is 0.356 on the expanded scope with a permutation null
far below it, but under repeated held-out splitting only **kallikrein 1**
($\rho_{\text{median}} = 0.756$, 100% positive splits) and **furin**
($0.617$, 99%) are accepted. Five proteases are actively **not supported**, among
them kallikrein-related peptidase 2 ($\rho_{\text{median}} = -0.009$, 50% positive
splits — indistinguishable from noise) and KLK7 ($-0.238$, 11%). The remaining
**50** of 57 have no usable holdout at all. The validation tiers are 2 validated, 12
supported, 43 assumed.

**Link 3 fails.** In `3_01` the modelled hazard does not order measured degradation
rates: $\rho = -0.039$ (uniform, $p = 0.75$) and $-0.020$ (concentration,
$p = 0.87$) over 68 sequences. Peptide length alone predicts them better
($\rho = -0.274$) and with the **opposite** sign to the model's own length
dependence. The offset $\alpha$ fits, but fitting an offset proves nothing. Every
horizon quoted in minutes anywhere in the `4_` notebooks is therefore provisional.

**Discrimination across scaffolds fails.** In `3_02` no predictor leaves the
neighbourhood of AUC 0.5 on the ten annotated peptides, under any weighting, and
the length control does not separate them either. The matched pairs behave
differently and instructively: EFK17-W is predicted to survive **4.56×** longer
than EFK17 against HNE alone, but only **1.11×** against the whole panel. The
substitution changes the P1′ preference of one enzyme while forty-six others
continue to contribute hazard it does not touch, so the panel sum dilutes a real
effect to nothing.

**Link 6 fails at the settings the notebooks originally used.** With every fragment
counted, the aureolysin digest of LL-37 — described in the source as abolishing
activity immediately — is predicted **2.6× more potent** than the intact peptide,
because a four-residue product receives a finite MIC only 1.7× worse than the
parent's while inheriting the parent's molar concentration. A minimum-length guard
between 9 and 21 residues reproduces both published digests.

**The MIC model fails on Trp-substituted variants.** The sources report EFK17-W as
4–8× more potent than EFK17 and CAM-W as 3–12× more potent than CAM. The predictor
puts both variants **less** potent, 0.82× and 0.75× on the reference strain, and the
strain sweep removes the escape: across all 34 APEX strains EFK17-W is predicted
more potent on 10 (median fold 0.91) and CAM-W on 3 (median 0.68); on the species
the sources used the folds run 0.71 to 1.26. Trp substitution is exactly the
modification the proteolysis side handles well, and it is the modification the
activity side gets backwards.

#### The summary judgement

The component that ranks bonds **within** a peptide is supported by measurement.
The component that compares rates or potencies **across** peptides is not. Results
from the `4_` notebooks may be read as statements about where a peptide is attacked
and in what order; they may not be read as statements about how long one peptide
survives relative to another.

### Known defects, not yet corrected

These are recorded here, where the method is read, rather than in a separate file.

1. **Uncorrected permutation p-values.** `2_01`, `2_03`, `2_04` and `2_06` compute
   the naive proportion instead of $(1+k)/(1+B)$ and therefore report `p = 0.000`
   and `q = 0.000`, which are not attainable values. These q-values propagate into
   the `verdict` column of `2_07_model_panel.csv` and hence into every downstream
   notebook.
2. **No guard against degenerate split sets in `2_03` and `2_06`.** Cathepsin L and
   cathepsin D produced 100 identical splits, so their `fraction_positive_splits`
   of 1.00 carries no information about stability. Both happen to be classified
   `not_supported` on the strength of their median correlation, so the consequence
   is contained, but the statistic is reported as if it were meaningful.
3. **`4_01` uses the unguarded mixture rule.** It predates `4_03` and applies no
   minimum-length guard. Its median $\log_2$ potency loss of $-0.325$ — the finding
   that the first cut on average *raises* the potency sum — is an artefact of the
   bookkeeping, as is the comparison $\rho = +0.972$ between the `concentration` and
   `concentration_only` orderings. Both require recomputation.
4. **The guard does not transfer to short peptides.** The range 9–21 was calibrated
   on LL-37, a 37-residue peptide. On the 12–30 residue probe of `4_02`, a guard of
   15 drives the median retained potency of two tiers to zero and 21 drives all
   three. An absolute minimum length is the wrong form for the constraint; it should
   scale with the parent, and that change has not been made.
5. **Row dependence in the `3_03` null.** LL-37 and EFK17 share sequence
   (EFK17 = LL-37 residues 16–32) and aureolysin appears in three of the eight rows.
   The permutation null controls the bond labelling but not the dependence between
   rows, so $p = 0.0005$ is optimistic about how much independent information the
   eight rows carry.
6. **The slope $\beta$ is extrapolated from one enzyme.** It is kallikrein 1's
   fitted slope applied to all 57 proteases. No per-protease slope survived
   validation, so there is no better estimate, but the panel-wide constant is an
   assumption and not a measurement.
7. **The hazard is dominated by unvalidated enzymes.** In `4_01` the dominant
   protease is thrombin for 229 of 300 peptides (median hazard share 0.735), plasma
   kallikrein for 62 and factor XIIa for 9 — **all three in the `assumed` tier**.
   No peptide is dominated by an enzyme that passed held-out validation, so the
   validated part of the panel and the deployed part do not intersect.
8. **The subsite mask biases the model's argmax toward the termini.** `1_01`
   measures a full eight-subsite window scoring 0.65 nats **lower** than a partial
   one, because most subsite contributions are negative and a shorter sum is a
   larger score. `first_cut_distribution` takes a softmax over bonds within a
   peptide, so terminal bonds carry that advantage systematically. Per-subsite
   normalisation removes it but converts the log-odds into a quantity that cannot
   be exponentiated into an intensity, so it is recorded rather than corrected.
9. **Abundance is negatively correlated with matrix quality.** `0_04` finds
   $\rho = -0.495$ against ranking stability and $-0.466$ against cleavage count,
   so the concentration weighting amplifies the least reliable part of the panel.
   This is a property of the data, not a bug, but it means a concentration-weighted
   result is weighted toward the thinly characterised enzymes.
10. **Matrix-level agreement statistics understate disagreement about decisions.**
   `0_08` finds a median Jensen-Shannon shift of 0.03 bits changing the nominated
   bond in roughly half of all peptides. Any agreement reported per subsite,
   including everything in `0_07`, therefore reads as more reassuring than the
   decision-level truth.
11. **The exopeptidase analysis is not connected to the dynamics.** `1_02` treats
   exopeptidase trimming as a ladder and finds it competitive with internal
   cleavage, but whether exopeptidase events actually fire in the `4_01` and `4_02`
   simulations, and how often, has not been measured.

## Contents

The notebooks follow one line of reasoning and are numbered in the order it runs.

### `notebooks/0_data_analysis/` — defining the datasets

Two objects are audited in parallel, each first on its own, then extended, then
compared, and only at the end joined: the **cleavage environment** (MEROPS and the
independent sources) and the **kinetics** (BRENDA and its wider extraction). The
series ends with the datasets themselves, not with a verdict.

| Notebook | Subject | Output |
| --- | --- | --- |
| `0_01_merops_matrices.ipynb` | MEROPS as an object: cleavage counts, subsite specificity against the proteome background, ranking stability under resampling, redundancy across the panel | `0_01_merops_serum_panel_audit.csv` |
| `0_02_serum_peptidase_identity.ipynb` | Which peptidases circulate, their EC numbers, families and event geometry — the classes every later analysis is cut along | `0_02_serum_protease_geometry.csv` |
| `0_03_serum_peptidase_abundance.ipynb` | How much of each circulates, mass spectrometry against immunoassay, and why an abundance is a prior and not a concentration | `0_03_hpa_abundance_coverage.csv`, `0_03_hpa_abundance_per_protease.csv` |
| `0_04_merops_through_serum_classes.ipynb` | Whether matrix quality depends on catalytic family, geometry or circulating amount, and which peptidases cannot be carried forward | `0_04_class_quality.csv`, `0_04_rejections.csv`, `0_04_class_tests.csv` |
| `0_05_source_mmp_pics.ipynb` | The PICS dataset on its own terms, against the MEROPS matrices of the same nine metallopeptidases, with a bootstrap band and the two peptide libraries as strata | `0_05_dataset.csv`, `0_05_intersection.csv`, `0_05_profile_agreement.csv`, `0_05_profile_agreement_strata.csv`, `0_05_profile_agreement_bootstrap.csv`, `0_05_contribution.csv` |
| `0_06_source_dpp4_qpisa.ipynb` | The qPISA dataset on the same template, against the MEROPS matrix of DPP4, with a bootstrap band and depletion strength as strata | `0_06_dataset.csv`, `0_06_intersection.csv`, `0_06_profile_agreement.csv`, `0_06_profile_agreement_strata.csv`, `0_06_profile_agreement_bootstrap.csv`, `0_06_contribution.csv` |
| `0_07_sources_against_merops.ipynb` | The two independent sources on one set of axes: how much evidence each adds against how far each agrees, with one comparison worked through by hand | `0_07_source_comparison.csv` |
| `0_08_merops_baseline_vs_extended.ipynb` | Folds the independent windows into the MEROPS counts and measures the change at three levels: the matrix, the bond ranking, and the bond the model nominates | `0_08_matrix_shift.csv`, `0_08_score_shift.csv`, `0_08_information_gain.csv` |
| `0_09_brenda_kinetics.ipynb` | BRENDA on its own, with no relation to MEROPS: what is quantitative, what is sequence-resolved, what is window-resolved | `0_09_brenda_coverage.csv` |
| `0_10_brenda_extension.ipynb` | The wider peptidase extraction against the serum one, and what widening does to BRENDA's own coverage | `0_10_expanded_kinetics_integration.csv` |
| `0_11_reaction_conditions.ipynb` | Assay conditions recovered from free text, their distributions per peptidase, the outlier rules, a thermodynamic normalisation and the exclusion policy | `0_11_condition_coverage.csv`, `0_11_condition_per_protease.csv`, `0_11_outliers.csv`, `0_11_replicate_scatter.csv`, `0_11_normalisation.csv`, `0_11_condition_policy.csv`, `0_11_leakage_bound.csv` |
| `0_12_serum_halflife_labels.ipynb` | The degradation-rate labels for validation B, and why plasma is kept out of serum | `0_12_serum_halflife_coverage.csv` |
| `0_13_control_panel_labels.ipynb` | The biologist's control panel as an independent criterion, and which of its proteases can be scored at all | `0_13_control_panel_relevance.csv` |
| `0_14_dataset_definition.ipynb` | The three decision axes — specificity, quantitative kinetics, control-panel relevance — and which peptidases fail on which | `0_14_dataset_definition.csv`, `0_14_validation_level_status.csv` |
| `0_15_analysis_datasets.ipynb` | The join: the rows carrying both a scored window and an efficiency, under compatible conditions, written out as the datasets series 2 consumes | `0_15_dataset_baseline.csv`, `0_15_dataset_extended.csv`, `0_15_assembly_losses.csv`, `0_15_dataset_comparison.csv` |

### `notebooks/1_merops_features/` — metrics on the cleavage environment

| Notebook | Subject | Output |
| --- | --- | --- |
| `1_01_score_metrics_and_masks.ipynb` | What the window score measures, how the subsite mask biases it, and what scoring a protease under the wrong geometry costs | `1_01_score_distribution.csv`, `1_01_mask_effect.csv`, `1_01_geometry_effect.csv` |
| `1_02_exopeptidase_processivity.ipynb` | Exopeptidases as a ladder of truncations rather than a single event: how far it descends, what each rung costs, and whether trimming competes with an internal cut | `1_02_exopeptidase_processivity.csv` |

### `notebooks/2_merops_to_kinetics/` — the relation, and which peptidases survive it

Paired: each question is asked on the baseline dataset and then on the extended
one, so the cost of the expansion is visible rather than absorbed.

| Notebook | Subject | Output |
| --- | --- | --- |
| `2_01_relation_baseline.ipynb` | Does the window score order measured efficiency, on the baseline dataset | `2_01_merops_vs_kinetics_per_protease.csv`, `2_01_dpp4_holdout.csv` |
| `2_02_holdout_baseline.ipynb` | Repeated cluster-held-out validation per protease, baseline | `2_02_per_protease_verdicts.csv`, `2_02_substrate_splits.csv`, `2_02_expansion_per_protease.csv`, `2_02_selected_protease_properties.csv`, `2_02_substrate_set_expansion.csv` |
| `2_03_relation_extended.ipynb` | The same relation on the extended dataset, with the two scopes compared | `2_03_expanded_panel_relation.csv`, `2_03_scope_comparison.csv` |
| `2_04_holdout_extended.ipynb` | Repeated held-out validation per protease, extended, marking which proteases the extension brought in and how the verdicts of the rest moved | `2_04_per_protease_verdicts.csv`, `2_04_substrate_splits.csv`, `2_04_extension_effect.csv` |
| `2_05_site_recovery_baseline.ipynb` | Whether the implied scissile bond is the reported one, baseline | `2_05_site_recovery_validation.csv`, `2_05_substrate_table.csv` |
| `2_06_site_recovery_extended.ipynb` | The same on the extended dataset | `2_06_site_recovery_validation.csv`, `2_06_substrate_table.csv` |
| `2_07_model_panel.ipynb` | The protease list the time-course analyses consume, with the panel's abundance coloured by validation verdict | `2_07_model_panel.csv`, `2_07_abundance_by_verdict.csv` |
| `2_08_score_window_variants.ipynb` | Whether a restricted subsite window — non-prime only, prime only, the core, P1 alone — predicts the measured rate better than the full one, on both datasets | `2_08_window_variants.csv`, `2_08_variant_per_protease.csv`, `2_08_variant_paired.csv`, `2_08_variant_bond_separation.csv` |

### `notebooks/3_serum_halflife/` — the space in which events happen

| Notebook | Subject | Output |
| --- | --- | --- |
| `3_01_time_axis_calibration.ipynb` | Fits the global log-intensity offset against measured serum half-lives, and tests separately whether the modelled hazard orders those measurements | `3_01_time_axis_calibration.csv`, `3_01_half_life_predictions.csv`, `3_01_length_control.csv` |
| `3_02_published_site_recovery.ipynb` | Whether the score puts the published cleavage sites where the experiments found them, for single enzymes and for the assembled panel | `3_02_single_enzyme_sites.csv`, `3_02_panel_sites.csv`, `3_02_site_null.csv` |
| `3_03_control_panel_discrimination.ipynb` | Whether the metrics separate peptides whose serum fate is annotated, and whether they move in the published direction on matched engineered pairs | `3_03_control_discrimination.csv`, `3_03_matched_pairs.csv`, `3_03_separation_summary.csv` |
| `3_04_discrimination_diagnostics.ipynb` | Whether the negative result is a defect, a power limit or a real absence: a power curve, a planted-signal positive control, and thirteen alternative predictors against a family-wise null | `3_04_power.csv`, `3_04_positive_control.csv`, `3_04_alternative_predictors.csv` |

### `notebooks/4_cut_to_mic/` — the consequences of cleavage

| Notebook | Subject | Output |
| --- | --- | --- |
| `4_01_single_cut_landscape.ipynb` | The exact distribution of the first cleavage under three weightings, with the potency each candidate cut leaves behind | `4_01_single_cut_landscape.csv`, `4_01_dominant_proteases.csv` |
| `4_02_activity_budget.ipynb` | Gillespie simulation over a probe stratified by potency, the distribution of the outcome across cleavage patterns, every transform over every peptide, and where the surviving potency sits for the extremes | `4_02_potency_distribution.csv`, `4_02_transform_comparison.csv`, `4_02_guard_sensitivity.csv`, `4_02_fragment_decomposition.csv` |
| `4_03_published_fragment_outcomes.ipynb` | Whether the predicted potency of the published cleavage products matches the reported outcome, and what minimum fragment length the mixture rule needs | `4_03_fragment_outcomes.csv`, `4_03_length_control.csv`, `4_03_mixture_rule.csv`, `4_03_engineered_pairs.csv`, `4_03_engineered_strain_sweep.csv` |
| `4_04_labelled_control_dynamics.ipynb` | The annotated control peptides through all four links on one set of molecules: hazard, the exact single cut, the effective MIC minute by minute, and how the killing power accumulates under every transform | `4_04_control_hazard.csv`, `4_04_control_single_cut.csv`, `4_04_control_trajectories.csv`, `4_04_control_budgets.csv` |
| `4_05_worked_examples.ipynb` | The annotated peptides over a full day, molecule by molecule: where the cuts fall on the sequence, how many molecules survive, and how much activity is left after 1, 4 and 24 hours — reported in minutes and per cent, with no rank statistic | `4_05_survival.csv`, `4_05_checkpoints.csv`, `4_05_cut_sites.csv`, `4_05_plain_units.csv`, `4_05_group_comparison.csv` |

### Parameters and dataset filters

Every notebook carries one cell, marked **Parameters**, holding its thresholds,
filters, sample sizes and seeds. Policy several notebooks obey lives in
`_analysis_common.py` and is imported rather than repeated: the assay-condition
window `TEMPERATURE_POLICY_C` and `PH_POLICY`, the minimum evidence per protease,
the split and permutation counts, the score slope, the minimum active fragment
length, the reference strain and the seed, together with
`apply_condition_policy()`. Changing a value there and re-running `0_15` changes
every dataset downstream, because nothing after `0_15` rebuilds the join.

### `notebooks_deprecated/`

The notebooks as they stood before the reordering, kept as a reference for the
numbering change. They are not maintained.

### `notebooks/_analysis_common.py`

Repository paths, figure style, the source-availability helper and the result
writer shared by the notebooks. It contains no analysis logic. Reusable metrics
are library components, listed below.

## Execution

The notebooks read precomputed artifacts and run in seconds to a few minutes.
There is no precompute step in this directory. `DEVICE` selects CUDA when it is
available; the environment holds `torch 2.5.1+cu118`, installed with

```shell
uv sync --extra cu118
```

The `cu118` extra is the one that supports the local GPU, whose compute
capability is 6.1.

```shell
# repository root
PATH=.venv/bin:$PATH JUPYTER_PATH=.venv/share/jupyter \
  .venv/bin/python3 -m jupyter nbconvert --to notebook --inplace --execute \
  assets/experiments/analysis/09_26/20_serum_proteolysis_calibration_max/notebooks/0_data_analysis/*.ipynb
```

## Library components used

Metrics are implemented in
`pep_compass.optimization.components.helpers.proteolysis` and imported by the
notebooks:

| Module | Responsibility |
| --- | --- |
| `background.py` | Reference residue distributions and the FASTA composition parser |
| `specificity.py` | Subsite divergence, its sampling floor, matrix resampling, panel redundancy, dataset lookup per peptidase code |
| `geometry.py` | Event geometry, admissible cut positions, subsite masks, masked window scoring |
| `substrates.py` | Parsing substrate descriptions, inferring the scissile bond, the reporter-substrate window |
| `dynamics.py` | The Markov jump process over fragments: admissible events with their intensities, the exact first-cut distribution, and Gillespie simulation |
| `activity/transforms.py` | The potency sum of a fragment mixture and the registered MIC-to-activity transforms |

Tests: `tests/optimization/components/helpers/test_proteolysis_*.py`.
