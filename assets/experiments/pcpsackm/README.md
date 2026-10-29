# pcpsackm

Protease-conditioned peptide stability and cleavage kinetics modeling.

Analyses in this project part are executed on datasets produced by the
[proteolysis data manager](../../tools/proteolysis_data_manager/docs/README.md). Notebooks read the
data under `data/proteolysis/` and use the repository `.venv` kernel. Dataset definitions, units and
commands are documented in the library and are not repeated here.

## Contents

### datasets_analysis

Inspection of the acquired and integrated datasets. Notebook order within a folder is the intended
reading order.

#### half_time

Contract: [half-life workflow](../../tools/proteolysis_data_manager/docs/HALF_LIFE_WORKFLOW.md),
[data dictionary](../../tools/proteolysis_data_manager/docs/HALF_LIFE_DATA_DICTIONARY.md).

| Notebook | Content |
|---|---|
| [0_01_acqusition_analysis](datasets_analysis/half_time/0_01_acqusition_analysis.ipynb) | Raw files of PEPlife2, PepMSND, DRAMP and Cavaco |
| [0_02_processed_dataset](datasets_analysis/half_time/0_02_processed_dataset.ipynb) | Missing metadata, source overlap, conditions and half-life distributions of the processed TSV tables |
| [0_03_estimation_interval_analysis](datasets_analysis/half_time/0_03_estimation_interval_analysis.ipynb) | Confidence-interval shape estimated from a half-life point; leave-one-peptide-out validation on 16 Cavaco peptides |

#### kinetics_data

Contract: [kinetics workflow](../../tools/proteolysis_data_manager/docs/KINETICS_WORKFLOW.md),
[kinetics contract](../../tools/proteolysis_data_manager/docs/KINETICS_CONTRACT.md).

| Notebook | Content |
|---|---|
| [acquisition_analysis](datasets_analysis/kinetics_data/acquisition_analysis.ipynb) | Original BRENDA and SABIO-RK records |
| [0_01_kinetics_coverage](datasets_analysis/kinetics_data/0_01_kinetics_coverage.ipynb) | Peptidase and parameter coverage of both sources |

#### window_neighbourhood

Contract: [window-neighbourhood workflow](../../tools/proteolysis_data_manager/docs/WINDOW_NEIGHBOURHOOD_WORKFLOW.md).

| Notebook | Content |
|---|---|
| [acquisition_analysis](datasets_analysis/window_neighbourhood/acquisition_analysis.ipynb) | Original MEROPS, PICS and qPISA files |
| [0_01_merops_coverage](datasets_analysis/window_neighbourhood/0_01_merops_coverage.ipynb) | MEROPS evidence counts and positional coverage |
| [0_02_merops_rank_stability](datasets_analysis/window_neighbourhood/0_02_merops_rank_stability.ipynb) | Stability of peptide rankings under count resampling |
| [0_03_mmp_pics_windows](datasets_analysis/window_neighbourhood/0_03_mmp_pics_windows.ipynb) | MMP/PICS source windows and P4–P4′ count matrices |
| [0_04_mmp_pics_qf_measurements](datasets_analysis/window_neighbourhood/0_04_mmp_pics_qf_measurements.ipynb) | QF coefficients and relative activities linked to cut windows |
| [0_05_mmp_pics_defined_sites](datasets_analysis/window_neighbourhood/0_05_mmp_pics_defined_sites.ipynb) | Synthetic QF cut sites and motif frequencies in the PICS libraries |
| [0_06_qpisa_endpoint_measurements](datasets_analysis/window_neighbourhood/0_06_qpisa_endpoint_measurements.ipynb) | qPISA EV1/EV4 endpoint measurements with substrate windows |
| [0_07_qpisa_timecourse_measurements](datasets_analysis/window_neighbourhood/0_07_qpisa_timecourse_measurements.ipynb) | qPISA EV6 substrate–product time courses |

#### test_models

Validation of published, pretrained half-life regressors on our own measurements, with
its own environment and artifacts: [module README](datasets_analysis/test_models/README.md).
Nothing is trained in the evaluation path; the one refit experiment is labelled as such.

| Notebook | Content |
|---|---|
| [0_01_evaluation_setup](datasets_analysis/test_models/notebooks/0_01_evaluation_setup.ipynb) | Which published models distribute weights, evaluation cohorts, measurement interval, training overlap, measured substitution pairs |
| [1_01 … 1_08](datasets_analysis/test_models/notebooks/) | One published PeptiVerse artifact each: provenance, batch interface, regression, interval and class accuracy, cost, substitution sensitivity |
| [2_01_results_overview](datasets_analysis/test_models/notebooks/2_01_results_overview.ipynb) | All artifacts side by side, and the cost of the two defects found in the distributed release |
| [2_02_substitution_sensitivity](datasets_analysis/test_models/notebooks/2_02_substitution_sensitivity.ipynb) | Cross-model single-substitution analysis and its consequence for optimisation |
| [3_01_refit_on_our_cohort](datasets_analysis/test_models/notebooks/3_01_refit_on_our_cohort.ipynb) | Published estimator families refitted on our cohort under a sequence-similarity split |

#### merge

Contract: [unified contract](../../tools/proteolysis_data_manager/docs/UNIFIED_CONTRACT.md).
Both notebooks operate in memory and write no CSV.

| Notebook | Content |
|---|---|
| [0_01_common_parameters](datasets_analysis/merge/0_01_common_parameters.ipynb) | Common numeric columns of `to_frame()` and availability by source |
| [0_02_position_probabilities](datasets_analysis/merge/0_02_position_probabilities.ipynb) | Positional amino-acid probability matrices from MEROPS, PICS and qPISA |

### published_model_analysis

Empty directory.
