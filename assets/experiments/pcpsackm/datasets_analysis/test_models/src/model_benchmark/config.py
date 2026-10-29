"""Shared configuration for the published half-life model evaluation.

All evaluation notebooks and scripts in this module read their paths, filters
and benchmark sizes from here, so that a single definition of the evaluation
set is used everywhere.
"""

from __future__ import annotations

from pathlib import Path

# Repository layout
## REMARK: The module lives below assets/experiments and is not installed, so
## the repository root is located by walking up to the processed data tree.
MODULE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = next(
    parent
    for parent in MODULE_ROOT.parents
    if (parent / "data/proteolysis/half_time_datasets/processed").is_dir()
)
PROCESSED_DATA = REPOSITORY_ROOT / "data/proteolysis/half_time_datasets/processed"
DATA_MANAGER_SOURCE = REPOSITORY_ROOT / "assets/tools/proteolysis_data_manager/src"

# Third-party artifacts and module outputs
ARTIFACTS_DIR = MODULE_ROOT / "artifacts"
PEPTIVERSE_DIR = ARTIFACTS_DIR / "peptiverse"
RESULTS_DIR = MODULE_ROOT / "results"

# Evaluation set definition
CANONICAL_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 2
MAX_LENGTH = 49

## REMARK: Named cohorts, from the condition requested for this study to the
## widest human cohort. The primary cohort fixes species, matrix and context,
## because a published serum predictor cannot be held responsible for in vivo
## pharmacokinetics or for a non-human matrix.
COHORTS = {
    "human_serum_in_vitro": {
        "species": ("human",),
        "matrix": ("serum",),
        "in_vivo_in_vitro": ("in vitro",),
        "description": "Human serum, in vitro: the condition published serum predictors target",
    },
    "human_serum_any": {
        "species": ("human",),
        "matrix": ("serum",),
        "in_vivo_in_vitro": None,
        "description": "Human serum, any measurement context",
    },
    "human_serum_plasma_in_vitro": {
        "species": ("human",),
        "matrix": ("serum", "plasma"),
        "in_vivo_in_vitro": ("in vitro",),
        "description": "Human serum and plasma, in vitro",
    },
    ## REMARK: The extended cohort drops the species restriction but keeps
    ## in vitro blood, because a published blood-stability predictor is trained
    ## on mammalian blood proteolysis. In vivo records are excluded everywhere:
    ## an in vivo half-life is pharmacokinetic clearance, not proteolysis.
    "blood_in_vitro_any_species": {
        "species": None,
        "matrix": ("serum", "plasma"),
        "in_vivo_in_vitro": ("in vitro",),
        "description": "Serum and plasma of any species, in vitro: largest proteolysis cohort",
    },
}
PRIMARY_COHORT = "human_serum_in_vitro"
## REMARK: Used where statistical power matters more than an exactly matched
## assay condition, such as the measured substitution pairs.
EXTENDED_COHORT = "blood_in_vitro_any_species"

# Numerical-cost benchmark
COST_BATCH_SIZES = (1, 10, 100, 1000)
COST_REPEATS = 3
COST_WARMUP_CALLS = 1
## REMARK: Synthetic cost sequences use the median length of the primary
## cohort so that measured throughput reflects our own peptide population.
COST_SEQUENCE_LENGTH = 13

# Sensitivity analysis
SENSITIVITY_SEED = 1986
SENSITIVITY_PEPTIDE_COUNT = 24
PERMUTATION_REPEATS = 10

# Duration classes are defined by the data contract, never redefined here.
CLASS_ORDER = ("short", "medium", "long")
