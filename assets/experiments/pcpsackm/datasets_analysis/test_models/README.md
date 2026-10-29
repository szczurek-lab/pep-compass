# test_models

Validation of **published, pretrained** peptide half-life regressors on our own
measurements. Nothing in the evaluation path is trained: each model is loaded exactly as
its authors distribute it and asked to predict the half-life of peptides from our
processed half-life dataset. The one exception is
[`refit.py`](src/model_benchmark/refit.py), which deliberately refits the published
estimator families on our cohort to separate a wrong architecture from a wrong training
set; its results are labelled as refits everywhere.

The question is narrow and practical: **can any published model serve as a stability
oracle in pep-compass, or does the model have to be built from scratch?**

## Availability audit

A published accuracy figure can only be checked when the trained parameters are
distributed. Of the half-life regressors in the literature review, exactly one
distributes its weights.

| Published model | Target | Artifact availability | Decision |
|---|---|---|---|
| PeptiVerse (Chatterjee Lab, 2026) | serum half-life, hours | Full weights on Hugging Face (`ChatterjeeLab/PeptiVerse`), 8 half-life artifacts | **Evaluated locally** |
| PlifePred (Mathur et al. 2018) | half-life in blood, seconds | No weights or code; the authors run a batch-submission service | **Evaluated over HTTP** |
| HLP (Sharma et al. 2014) | half-life in intestine-like environment | Web server only | Not evaluated: the operator returns `403: Submissions from external sources or automated bots are blocked`, and `robots.txt` disallows `/cgibin/` for all agents. The matrix is also intestinal, not serum |
| PepMSND (Hu et al. 2025) | blood stability as a threshold label | Web server only | Not evaluated: `model.highslab.com` did not respond over HTTP or HTTPS. The target is a threshold label, and its 635 training records are a source of our own dataset |
| Tan et al. 2024 | half-life across species and organs | No code or weight release located | Not evaluated |
| Cavaco et al. 2021 | serum half-life from physicochemical properties | Publisher page returned HTTP 403, so the published coefficients were not retrieved | Not evaluated |

The same list is encoded in
[`predictors/__init__.py`](src/model_benchmark/predictors/__init__.py) as
`UNAVAILABLE_MODELS`, and the observed evidence for each blocked service in
[`predictors/webservice.py`](src/model_benchmark/predictors/webservice.py) as
`BLOCKED_SERVICES`, so the absence of a benchmark number stays visible in the result
tables together with its reason.

### Querying a published service

PlifePred distributes nothing, but its authors advertise a batch-submission module, so the
published model can still be measured. The policy this module applies, decided before any
request was sent:

* the endpoint is not excluded by `https://webs.iiitd.edu.in/robots.txt`, whose wildcard
  group requests `Crawl-delay: 20`, and the adapter waits that long between requests;
* peptides go out in batches of 64, so one request covers many sequences;
* every response is cached on disk, so a repeated evaluation sends nothing;
* the synthetic timing benchmark is not run against the service, and its substitution scan
  is reduced to two short peptides, because a full scan would cost thousands of requests
  to someone else's server;
* each result row is cross-checked against the submitted sequence, so a shifted result
  table is detected rather than silently mislabelled.

HLP is the counter-example: it is not queried at all, because its operator explicitly
blocks automated submission.

### The eight evaluated artifacts

PeptiVerse ships two estimator families over four input representations:

| Predictor key | Artifact directory | Estimator | Representation | Target |
|---|---|---|---|---|
| `peptiverse_transformer_esm_log` | `transformer_wt_log` | transformer head | ESM-2 650M, sequence | `log1p(hours)` |
| `peptiverse_transformer_esm_raw` | `transformer_wt_raw` | transformer head | ESM-2 650M, sequence | hours |
| `peptiverse_xgb_esm_log` | `xgb_wt_log` | XGBoost | ESM-2 650M, pooled | `log1p(hours)` |
| `peptiverse_xgb_esm_raw` | `xgb_wt_raw` | XGBoost | ESM-2 650M, pooled | hours |
| `peptiverse_transformer_peptideclm` | `transformer_peptideclm` | transformer head | PeptideCLM-23M, SMILES | hours |
| `peptiverse_xgb_peptideclm` | `xgb_peptideclm` | XGBoost | PeptideCLM-23M, SMILES | hours |
| `peptiverse_transformer_chemberta` | `transformer_chemberta` | transformer head | ChemBERTa-77M, SMILES | hours |
| `peptiverse_xgb_chemberta` | `xgb_chemberta` | XGBoost | ChemBERTa-77M, SMILES | hours |

The authors report that their half-life models were trained on **130 sequence-based** and
**245 SMILES-based** records.

## Evaluation cohort

The cohort rule reuses the data contract rather than inventing one:
`regression_label_available` (exact half-life relation, positive point estimate), the
canonical amino-acid alphabet, and lengths from 2 to 49 residues. Duration classes come
from `proteolysis_data_manager.integration.half_life.labels` (short below 30 min, medium
up to 6 h, long above).

| Cohort | Definition | Peptides | Baseline MAE log10 | Majority class | Measured substitution pairs |
|---|---|---|---|---|---|
| `human_serum_in_vitro` (primary) | human, serum, in vitro | 128 | 0.676 | 0.430 | 12 (9 resolvable) |
| `human_serum_any` | human, serum, any context | 142 | — | — | — |
| `human_serum_plasma_in_vitro` | human, serum and plasma, in vitro | 252 | 0.773 | — | — |
| `blood_in_vitro_any_species` (extended) | serum and plasma, any species, in vitro | 391 | 0.806 | 0.386 | 49 (35 resolvable) |

The extended cohort exists because the primary one is small. It triples the peptide count
by accepting rat, mouse, bovine and dog blood next to human blood, and it quadruples the
measured single-substitution pairs, which is the part of the analysis most limited by
sample size. **In vivo records are excluded from every cohort**: an in vivo half-life
measures pharmacokinetic clearance, not proteolytic stability.

Repeated measurements of one peptide are collapsed to their geometric mean, because a
sequence-only predictor cannot distinguish repeated conditions. The within-peptide spread
is kept as the measurement noise floor against which model errors are judged.

## What is measured

### Accuracy, in three increasingly easy forms

1. **Point regression** on `log10` seconds: MAE, RMSE, R², Pearson, Spearman, bias and the
   median error factor (`10**MAE`, read as "wrong by a factor of"). Every figure is
   printed next to the constant predictor that always answers the cohort median.
2. **Agreement with the measurement interval.** The logarithmic relative-width estimator
   from [`0_03_estimation_interval_analysis`](../half_time/0_03_estimation_interval_analysis.ipynb)
   turns a measured point into lower and upper limits; a prediction inside them counts as
   a hit. *Caveat, reported with every hit rate:* that estimator is fitted on 16 published
   Cavaco intervals in 50% human serum over a 8–322 min domain, so most of our cohort is
   scored by extrapolation.
3. **Duration class**, against the majority-class baseline.

Artifacts that ship a conformal calibration are additionally scored on the empirical
coverage of their own interval.

### Training overlap

The PeptiVerse training table is distributed as SMILES, so the overlap with our cohort is
determined exactly by matching canonical structures (RDKit InChIKey), not guessed from
data-source names. Accuracy is split into the overlapping part, which reports the model's
fit to its own training data, and the held-out part, which is the only transfer estimate.

### Numerical cost

Latency of a single peptide, throughput at batch sizes 1, 10, 100 and 1000, peak GPU
memory, and the projection to 1 and 1000 peptides. Both calling patterns are measured: the
published one peptide per call, and the batched path of this module.

### Sensitivity to one substitution

1. **Complete substitution scan**: every residue replaced by all 19 alternatives, giving
   the response size and whether it depends on the edited position.
2. **Permutation test**: the same composition reordered. A composition-only representation
   moves under substitution but not under reordering, so the ratio of the two responses
   separates a positional model from a composition-only one.
3. **Input gradients** for the differentiable artifacts, checked against the exhaustive
   scan: a gradient that ranks positions differently from the model's own predictions
   cannot drive a search.
4. **Measured parent-variant pairs** at Hamming distance one within one assay condition.
   The primary cohort yields 12 such pairs, which can refute a model but cannot certify
   one.

## Adapter verification

Negative results are only informative if the adapter reproduces the published forward
path, so this is established before any of them is read. `scripts/compare_with_upstream.py`
runs the authors' own `inference.py` from the snapshot and compares predictions peptide by
peptide. Under the deployed output convention both routes agree to **4.6e-16 relative
difference**, that is to floating-point precision, for the ESM sequence artifact and the
PeptideCLM SMILES artifact that their manifest selects. `scripts/validate_adapter.py`
additionally confirms that the batched path equals single-peptide calls for every artifact
(largest deviation 1.1e-6 in `log10` seconds, and 5.0e-4 for `xgb_chemberta`, which is
batched-GEMM float noise).

## Defects found in the distributed artifacts

Four defects were found while building the adapters. Each is a property of the upstream
release, verified against the authors' own code, not an artefact of our evaluation.

1. **The published inference script never inverts the `log1p` target transform, so the
   deployed predictor reports `log1p(hours)` as hours.** The branch is guarded by
   `"log" in model_name`, where `model_name` is the manifest alias (`transformer`, `xgb`)
   rather than the artifact folder (`transformer_wt_log`), so it is unreachable. Verified
   numerically: for `ALAGIGILTV` the upstream script returns 1.9877 "hours", which is
   exactly `log1p(6.2984)`, the value the artifact actually predicts. Because the
   transform is monotone, rank correlation is unaffected, but the error is not: MAE rises
   from 0.650 to 0.704 `log10` and class accuracy falls from 0.445 to 0.391.
2. **The feature scaler shipped with the XGBoost sequence artifacts is never applied**,
   and applying it changes the outcome decisively. `scaler.joblib` sits in `xgb_wt_log`
   and `xgb_wt_raw` and no code path loads it; applying it moves predictions by 1.06
   `log10` (median) and lifts the two weakest artifacts to the level of the best ones:
   on the primary cohort Spearman goes from 0.031 to 0.251 and from 0.011 to 0.292, and
   class accuracy from 0.312 to 0.469 and from 0.320 to 0.477, the highest in that table.
   On the extended cohort the same correction takes `xgb_esm_log` from MAE 0.882 to 0.782,
   that is from above to below the trivial baseline. The artifacts were evidently fitted on
   scaled features, so the distributed deployment discards much of their skill.
3. **The conformal calibration of the ESM-based artifacts cannot be loaded by the authors'
   own script.** The bundles are pickled MAPIE objects, while `inference.py` requires a
   dictionary with a `quantile` key and otherwise raises `Unrecognised MAPIE bundle`; the
   failure was observed directly when running their code. This module recovers the
   half-width from the stored conformity scores instead.
4. **The conformal half-width is applied in the wrong space.** It was fitted in the
   training target space, but the published script adds it to the back-transformed
   prediction. Both readings are reported; neither is usable in practice, since the
   recovered half-width of 1.86 in `log1p(hours)` corresponds to a multiplicative interval
   of roughly five orders of magnitude.

Additionally, the calibration was computed on 130 conformity scores, which is the size of
the whole training set, so those intervals are in-sample and their nominal 90% level is
not a validated coverage guarantee.

## Results

Primary cohort: human serum, in vitro, 128 peptides. The constant predictor that always
answers the cohort median has **MAE 0.676 `log10`**, and the majority duration class
covers **43%** of the cohort. Both are the references every figure must be read against.

| Artifact (as published) | MAE log10 | R² | Spearman | Class accuracy | In measurement interval | Spearman, held out | Peptides/s |
|---|---|---|---|---|---|---|---|
| `transformer_esm_log` | **0.650** | 0.004 | **0.300** | 0.445 | 0.227 | 0.307 | 79 |
| `transformer_esm_raw` | 0.656 | 0.026 | 0.216 | 0.430 | 0.203 | 0.239 | 79 |
| `xgb_peptideclm` | 0.680 | −0.031 | 0.056 | 0.430 | 0.203 | 0.033 | 138 |
| `xgb_chemberta` | 0.697 | −0.078 | 0.073 | 0.422 | 0.227 | 0.017 | **519** |
| `transformer_peptideclm` | 0.714 | −0.106 | 0.144 | 0.398 | 0.148 | 0.092 | 130 |
| `transformer_chemberta` | 0.732 | −0.154 | −0.085 | 0.422 | 0.180 | −0.155 | 334 |
| `xgb_esm_log` | 0.865 | −0.659 | 0.031 | 0.312 | 0.062 | 0.054 | 79 |
| `xgb_esm_raw` | 0.881 | −0.739 | 0.011 | 0.320 | 0.227 | −0.027 | 78 |
| `xgb_esm_log` + shipped scaler | 0.668 | −0.051 | 0.251 | 0.469 | 0.258 | — | 79 |
| `xgb_esm_raw` + shipped scaler | 0.684 | −0.226 | 0.292 | **0.477** | 0.289 | — | 78 |

PlifePred, queried through its authors' batch service, is in the table below for the
extended cohort; on the primary cohort it reaches MAE 0.884, R² −0.864, Spearman 0.222 and
class accuracy 0.320.

Reading of the table:

* **No artifact beats the trivial baseline in error.** Only `transformer_esm_log` is below
  it at all, by 0.026 `log10`, and six of eight have negative R², meaning they are worse
  than answering the cohort median.
* **Ranking is weak everywhere.** The best Spearman is 0.300, one artifact is negative,
  and removing the fifth of the cohort that appears in the published training set does not
  rescue any of them.
* **Class assignment does not beat guessing the majority class** (0.43) except marginally,
  and only for the two scaler-corrected runs.
* **The interval test fails badly.** At best 23% of predictions land inside the measurement
  interval estimated around the measured point, and 37% among the peptides inside the
  estimator's fitted domain. The band is not tight but not trivial either: its median width
  is a factor of 2.8. Note that 42% of the cohort lies outside the fitted domain and is
  scored by extrapolation.
* **Cost is not the obstacle.** 79 peptides per second with ESM-2 650M on a GTX 1060, up to
  519 with ChemBERTa, which is 13 s and 2 s respectively for a thousand candidates. Batching
  the published per-peptide path gives a 1.5 to 3.9 times speed-up.

### The same question on three times as many peptides

Extended cohort: serum and plasma of any species, in vitro, 391 peptides. Its constant
baseline is **MAE 0.806 `log10`** and its majority class covers **38.6%**.

| Run (as published) | MAE log10 | R² | Spearman | Class accuracy | In measurement interval |
|---|---|---|---|---|---|
| `transformer_esm_log` | **0.741** | 0.075 | **0.384** | 0.409 | 0.174 |
| `transformer_esm_raw` | 0.780 | 0.040 | 0.291 | 0.345 | 0.171 |
| `xgb_peptideclm` | 0.812 | −0.012 | 0.056 | 0.338 | 0.136 |
| `transformer_chemberta` | 0.829 | −0.062 | 0.029 | 0.340 | 0.166 |
| `xgb_chemberta` | 0.835 | −0.070 | −0.124 | 0.325 | 0.148 |
| `transformer_peptideclm` | 0.845 | −0.086 | −0.065 | 0.327 | 0.102 |
| `xgb_esm_log` | 0.882 | −0.258 | 0.291 | 0.407 | 0.097 |
| `plifepred_svm` (web service) | 0.906 | −0.455 | 0.167 | **0.430** | 0.182 |
| `xgb_esm_raw` | 0.997 | −0.649 | 0.261 | 0.299 | 0.189 |

Tripling the cohort does not change the conclusion, it sharpens it:

* **Two of nine runs beat the trivial baseline in error**, both of them the ESM transformer
  heads, and by 0.03 to 0.07 `log10`. Seven remain worse than answering the cohort median.
* Ranking improves slightly for the ESM artifacts, from 0.300 to 0.384 for the best one,
  which is the expected effect of a wider target range rather than better prediction: R²
  stays at 0.075.
* The SMILES artifacts stay at or below zero correlation on both cohorts, so their result
  is not a small-sample artefact.
* `plifepred_svm` has the highest class accuracy of the whole table at 0.430, while
  predicting "short" for 90% of peptides. On a cohort whose majority class covers 38.6%
  that is an artefact of the class balance, not skill: its R² is the worst in the table.

### Sensitivity to one substitution

| Artifact | Median \|Δ\| substitution | Reordering / substitution | Direction agrees, measured pairs | Predicted / measured effect |
|---|---|---|---|---|
| `transformer_esm_log` | 0.101 | 1.17 | 0.67 | 0.14 |
| `transformer_esm_raw` | 0.101 | 1.06 | 0.44 | 0.27 |
| `transformer_chemberta` | 0.063 | 1.80 | 0.56 | 0.20 |
| `transformer_peptideclm` | 0.054 | 1.54 | 0.78 | 0.08 |
| `xgb_esm_log` | 0.049 | 0.98 | 0.33 | 0.10 |
| `xgb_chemberta` | 0.072 | 1.05 | 0.33 | 0.11 |
| `xgb_peptideclm` | 0.025 | 1.24 | 0.44 | 0.06 |
| `xgb_esm_raw` | 0.012 | 0.96 | 0.44 | 0.01 |

Every artifact does react to a single substitution, and the reaction does depend on
residue order. The problem is the size and the direction: predicted effects are **0.01 to
0.27 times the measured ones** (measured median 0.38 `log10`, a factor of 2.4), so a
search driven by these models would move in steps far smaller than the effects it is
supposed to find, and the direction agrees on only 33% to 78% of the 12 measured pairs,
which is not distinguishable from chance at that sample size.

On the extended cohort the empirical test has **35 pairs** instead of 9, and it stays
negative. The measured effects are larger there (median 0.426 `log10`, a factor of 2.7),
while the predicted ones remain 0.01 to 0.23 times that size:

| Run | Direction agrees | Spearman of Δ | Predicted / measured effect | Reordering / substitution |
|---|---|---|---|---|
| `xgb_esm_raw` | 0.600 | **0.327** | 0.040 | 1.20 |
| `transformer_esm_log` | 0.571 | 0.032 | 0.217 | 1.51 |
| `transformer_chemberta` | 0.571 | 0.106 | 0.112 | 2.92 |
| `xgb_esm_log` | 0.457 | −0.125 | 0.099 | 1.09 |
| `xgb_chemberta` | 0.429 | −0.042 | 0.092 | 1.71 |
| `transformer_esm_raw` | 0.400 | −0.132 | 0.228 | 1.13 |
| `transformer_peptideclm` | 0.400 | −0.079 | 0.219 | 1.98 |
| `xgb_peptideclm` | 0.371 | −0.188 | 0.075 | 1.58 |
| `plifepred_svm` | **0.171** | **−0.231** | 0.014 | **0.00** |

The best direction agreement is 0.60 on 35 pairs, which a coin flip reaches with
probability 0.15, so no artifact demonstrates that it predicts the effect of a
substitution. Four of nine are anti-correlated with the measurement.

**PlifePred is the clearest case.** Its reordering ratio is exactly **0.00**: all ten
permutations of every probe peptide return the identical prediction, while substitutions
move it by 0.13 `log10` on average. The model reads amino-acid composition only and cannot
distinguish a peptide from its anagrams. On the measured pairs its predicted direction is
right 17% of the time, that is reliably wrong, and its predicted effects are 1.4% of the
measured ones. Its dynamic range is also collapsed: on the primary cohort it answers
between 322 s and 5037 s, spanning 1.2 `log10` where the measurements span 3.5, so it
never predicts the `long` class at all and carries a bias of −0.76 `log10`, a systematic
factor of 5.8 too short. Its ranking on the 39 extended-cohort peptides that come from its
own training source is Spearman 0.696, against 0.139 on the 352 peptides it never saw,
which is the signature of a model recalled rather than generalised.

### Is the architecture or the training set at fault

`3_01_refit_on_our_cohort.ipynb` refits the same families on our cohorts with the backbone
frozen, under a sequence-similarity split (greedy Levenshtein clustering, grouped 5-fold).

| Estimator | Spearman, human serum (128 peptides, 33 clusters) | Spearman, serum and plasma (252 peptides, 86 clusters) |
|---|---|---|
| ridge on ESM-2 | **0.396** (MAE 0.725) | 0.164 (MAE 0.795) |
| gradient boosting | 0.106 | 0.032 |
| dense head | 0.155 | 0.055 |
| median baseline | — (MAE 0.676) | — (MAE 0.773) |

Two results, both negative in a useful way:

* A ridge head on frozen ESM-2 embeddings ranks our peptides better than any published
  artifact (0.396 against 0.300), yet **still does not beat the constant baseline in
  error**. The published training set is therefore not the only problem.
* **Doubling the sample by adding plasma makes ranking worse**, not better (0.396 to
  0.164). More data of this kind does not help, because the added measurements come from a
  different matrix. The binding constraint is the consistency of the measurement condition
  together with the representation, not the number of rows.

One positive detail: the refitted ridge head responds to a single substitution with a
median of 0.367 `log10`, which is the scale of the measured effects (0.383), whereas the
published artifacts respond with 0.059. A model fitted on our own condition reacts at the
right magnitude even when its absolute calibration is poor.

Taken together this supports the direction the literature review proposed, namely local
cleavage-site features with conditioning on the assay, rather than another global
whole-peptide sequence regressor.

## Conclusion

Counted over the nine published configurations, on both cohorts:

| | Primary cohort (128 peptides) | Extended cohort (391 peptides) |
|---|---|---|
| Beat the constant-median baseline in error | 2 of 9 | 2 of 9 |
| Order peptides better than chance (\|Spearman\| > 0.2) | 3 of 9 | 4 of 9 |
| Beat the majority duration class | 1 of 9 | 3 of 9 |
| Best model | `transformer_esm_log`, Spearman 0.300 | `transformer_esm_log`, Spearman 0.384 |

1. **No published half-life model is calibrated for our data.** The best one is 0.03 to
   0.07 `log10` better than answering the cohort median, with R² near zero, and most are
   worse than that baseline. None is usable as an absolute stability predictor.
2. **None demonstrates that it predicts the effect of a single substitution**, which is
   the operation pep-compass performs. On 35 measured pairs the best direction agreement is
   0.60, which chance reaches with probability 0.15, and four of nine are anti-correlated.
   Predicted effects are 1% to 23% of measured ones.
3. **PlifePred cannot see residue order at all** (permutation response exactly 0.00) and
   predicts within 1.2 `log10` where measurements span 3.5. It is disqualified for
   sequence optimisation by construction, not by accuracy.
4. **Part of the apparent weakness is a packaging defect, not a modelling one.** The
   deployed PeptiVerse predictor reports `log1p(hours)` as hours, and its XGBoost artifacts
   are served without the feature scaler they were fitted with; correcting the scaler moves
   `xgb_esm_log` from MAE 0.882 to 0.782 on the extended cohort. Both defects are verified
   against the authors' own code.
5. **Refitting does not rescue the approach.** The same estimator families on frozen ESM-2
   embeddings reach Spearman 0.396 under a similarity split, still without beating the
   constant baseline in error, and adding more heterogeneous data makes ranking worse.

The recommendation that follows is not to adopt or fine-tune any of these models, but to
build a stability oracle around local cleavage-site features with explicit conditioning on
the assay, and to validate it on parent-variant pairs rather than on a global R².

## Environment

The module keeps its own environment, isolated from the repository `.venv`:

* PeptiVerse inference pins `transformers==4.46.0`, and the calibration bundles need
  `mapie>=1.0`;
* the available GPU is a GTX 1060 (compute capability 6.1, Pascal), which needs a CUDA
  11.8 build, while the repository environment runs `torch 2.10.0+cpu`.

```bash
cd assets/experiments/pcpsackm/datasets_analysis/test_models
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt
```

`artifacts/` and `.venv/` are not tracked.

## How to run

```bash
# 1. Fetch the published artifacts and their representation backbones (about 3 GB)
.venv/bin/python scripts/download_peptiverse.py

# 2. Verify the adapter against the authors' own inference script before trusting results
.venv/bin/python scripts/validate_adapter.py
.venv/bin/python scripts/compare_with_upstream.py

# 3. Evaluate the downloaded artifacts on both cohorts
.venv/bin/python scripts/run_all_models.py --cohort human_serum_in_vitro
.venv/bin/python scripts/run_all_models.py --cohort blood_in_vitro_any_species

# 4. Measure the two release defects
.venv/bin/python scripts/run_variant_checks.py --cohort human_serum_in_vitro

# 5. Query the published web service, honouring its operator's crawl delay
#    (responses are cached, so a repeated run sends nothing)
.venv/bin/python scripts/run_all_models.py --models plifepred_svm --cohort human_serum_in_vitro

# 6. Generate and execute the notebooks from the stored results
.venv/bin/python scripts/make_notebooks.py --execute
```

`run_all_models.py --include-web-services` adds every registered service to a run. It is
off by default, so a plain run sends no external request.

Results are written per cohort and run to `results/<cohort>/<run>/`: `summary.json`,
`predictions.csv`, `confusion_matrix.csv`, `cost_batched.csv`, `cost_projection.csv`,
`mutational_scan.csv`, `scan_summary.csv`, `permutation_test.csv`,
`gradient_agreement.csv` and `measured_pairs.csv`. Adapter-verification reports sit at the
top of `results/`. The notebooks only read these files, so every notebook and the overview
report identical numbers.

## Notebooks

| Notebook | Content |
|---|---|
| `0_01_evaluation_setup.ipynb` | Availability audit, cohorts, measurement interval, training overlap, measured substitution pairs |
| `1_01` … `1_08_peptiverse_*.ipynb` | One published artifact each: provenance, batch-interface check, the three accuracy questions, training-overlap split, cost, sensitivity, verdict |
| `1_09_plifepred_svm.ipynb` | The same protocol for PlifePred, queried through its authors' batch service |
| `2_01_results_overview.ipynb` | All runs side by side: accuracy, overlap effect, the cost of the release defects, timing |
| `2_02_substitution_sensitivity.ipynb` | Cross-model single-substitution analysis and what it means for optimisation |
| `2_03_cohort_comparison.ipynb` | Every run on both cohorts: does the conclusion survive tripling the peptide count? |
| `3_01_refit_on_our_cohort.ipynb` | Published estimator families refitted on our cohorts under a sequence-similarity split |

## Module layout

```
src/model_benchmark/
  config.py        paths, cohort definitions, benchmark sizes
  dataset.py       cohort construction from the processed tables
  predictors/      batch interface, one adapter per published artifact, web-service adapters
  metrics.py       regression, interval and class scoring with baselines
  cost.py          latency, throughput and projections
  sensitivity.py   substitution scan, permutation test, gradients, measured pairs
  leakage.py       exact overlap with published training data
  refit.py         refits of the published families on our cohorts
  pipeline.py      per-model evaluation runs and persisted results
  report.py        tables and figures shared by the notebooks
scripts/
  download_peptiverse.py    selective download of the half-life artifacts
  validate_adapter.py       training-set recovery, batch equivalence, scaler effect
  compare_with_upstream.py  prediction-by-prediction check against the authors' script
  run_all_models.py         evaluation runs, optionally including web services
  run_variant_checks.py     the two release defects, measured
  make_notebooks.py         notebook generation and execution
```
