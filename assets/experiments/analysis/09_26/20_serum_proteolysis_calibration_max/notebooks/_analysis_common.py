"""Shared paths, plotting style and small loaders for the serum-calibration notebooks.

Every notebook in this analysis starts from the same repository paths, the same
figure style and the same source-status vocabulary, so this module exists to keep
one definition of each instead of one per notebook. It holds **no analysis
logic**: reusable metrics live in
``pep_compass.optimization.components.helpers.proteolysis``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import pandas as pd


def project_root() -> Path:
    """Locate the repository root from the current working directory.

    :return: Absolute repository root.
    :raises RuntimeError: If no ancestor directory contains ``src/pep_compass``.
    """

    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "src" / "pep_compass").is_dir():
            return candidate
    raise RuntimeError("Repository root with src/pep_compass not found above the working directory.")


def ensure_importable() -> Path:
    """Put the repository ``src`` directory on ``sys.path``.

    :return: Absolute repository root.
    """

    root = project_root()
    source_path = str(root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    return root


ROOT = ensure_importable()

# Data roots
MEROPS_ROOT = ROOT / "data" / "merops"
SERUM_ROOT = ROOT / "data" / "serum_proteolysis_calibration"
SERUM_MANIFEST = SERUM_ROOT / "manifests" / "serum_proteases.tsv"
SERUM_RAW = SERUM_ROOT / "raw"
SERUM_PROCESSED = SERUM_ROOT / "processed"
BRENDA_KINETICS = SERUM_PROCESSED / "kinetics" / "brenda_serum_proteases.parquet"
# Wider extraction covering every peptidase EC number BRENDA holds human kinetics
# for, produced by map_ec_to_merops.py followed by ingest_brenda.py --ec-mapping.
BRENDA_ALL_PEPTIDASES = SERUM_PROCESSED / "kinetics" / "brenda_all_peptidases.parquet"
PEPTIDES_APEX_CSV = ROOT / "data" / "peptides_data" / "peptides_mic" / "peptides_apex.csv"
DBAASP_FRAGMENTS_CSV = ROOT / "data" / "peptides_data" / "peptides_mic" / "dbaasp_fragments_mic.csv"
CONTROLS_MIC_CSV = ROOT / "data" / "peptides_data" / "peptides_mic" / "controls_mic.csv"

# Analysis outputs.
# REMARK: Derived from this file's location rather than from a hard-coded name,
# so renaming or renumbering the analysis directory does not break the notebooks.
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ANALYSIS_ROOT / "results"

# MEROPS dataset id of the human serum/plasma protease panel.
SERUM_DATASET_ID = 35
HUMAN_DATASET_ID = 34


# ---------------------------------------------------------------------------
# Shared analysis policy
#
# Everything here is a knob. A notebook that needs a different value overrides it
# in its own parameters cell; a change made here propagates to every notebook that
# does not override it, which is what makes the policy auditable in one place.
# ---------------------------------------------------------------------------

#: Assay conditions outside which a BRENDA measurement is not treated as an
#: observation of the same enzyme. Temperature is a rescaling within a moderate
#: range; pH changes ionisation and fold stability, so it is an exclusion.
TEMPERATURE_POLICY_C = (20.0, 40.0)
PH_POLICY = (6.0, 9.0)

#: Measurements of mutant enzymes describe a different protein from the one whose
#: matrix is being tested.
EXCLUDE_MUTANT_ENZYMES = False

#: Minimum evidence for a protease to enter a within-protease regression.
MIN_SUBSTRATES_PER_PROTEASE = 3
MIN_ROWS_FOR_CORRELATION = 4

#: Repeated held-out evaluation.
N_CLUSTER_SPLITS = 100
MIN_SUBSTRATES_PER_SPLIT_SIDE = 6
CLUSTER_SIMILARITY_THRESHOLD = 0.8

#: Permutation tests. The p-value is always (1 + k) / (1 + B).
N_PERMUTATIONS = 2000

#: The score-to-log-intensity coefficient, in nats. Fitted on kallikrein 1, the one
#: per-protease slope that survived held-out validation, and extrapolated.
SCORE_SLOPE_PER_DECADE = 0.404

#: Fragments shorter than this are treated as inactive rather than given a MIC the
#: predictor was not trained to produce. Calibrated in 4_03.
MIN_ACTIVE_FRAGMENT_LENGTH = 9

#: Reference strain for every MIC quoted in the analysis.
MIC_STRAIN = "A. baumannii ATCC 19606"

#: One seed for the whole analysis; per-protease generators derive from it.
SEED = 20260920


def apply_condition_policy(
    frame,
    temperature_column: str = "temperature_c",
    ph_column: str = "ph",
    temperature_policy: tuple[float, float] | None = None,
    ph_policy: tuple[float, float] | None = None,
    drop_missing: bool = False,
):
    """Filter kinetic measurements to biologically comparable assay conditions.

    Entries whose condition the source does not state are kept by default: a
    missing annotation is not evidence of an extreme condition, and discarding
    them would remove a third of the table on the strength of formatting.

    :param frame: Measurements carrying parsed conditions.
    :param temperature_column: Column holding the assay temperature in Celsius.
    :param ph_column: Column holding the assay pH.
    :param temperature_policy: Inclusive bounds, defaulting to the shared policy.
    :param ph_policy: Inclusive bounds, defaulting to the shared policy.
    :param drop_missing: Whether to also drop entries with no stated condition.
    :return: The filtered frame.
    """

    low_t, high_t = temperature_policy or TEMPERATURE_POLICY_C
    low_p, high_p = ph_policy or PH_POLICY
    temperature = frame[temperature_column]
    acidity = frame[ph_column]
    keep_t = temperature.between(low_t, high_t)
    keep_p = acidity.between(low_p, high_p)
    if not drop_missing:
        keep_t = keep_t | temperature.isna()
        keep_p = keep_p | acidity.isna()
    return frame[keep_t & keep_p]

# Categorical hues in fixed assignment order, validated for colour-vision
# deficiency separation against a light surface. Never cycle past the last slot;
# an additional group folds into ``NEUTRAL`` instead.
CATEGORICAL_COLORS = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")
NEUTRAL = "#6b6b6b"
GRID_COLOR = "#d9d9d9"
# Single-hue ramp for magnitude, two-hue ramp with a neutral midpoint for polarity.
SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"


def apply_style() -> None:
    """Apply the shared matplotlib style for this analysis.

    Axes stay recessive (light grid, no top/right spines) so the marks carry the
    figure.
    """

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.prop_cycle": mpl.cycler(color=list(CATEGORICAL_COLORS)),
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.size": 10,
        }
    )


def load_serum_manifest() -> pd.DataFrame:
    """Load the serum protease manifest as a table.

    :return: Manifest rows with ``include_in_hydrolysis_model`` as a boolean.
    :raises FileNotFoundError: If the manifest has not been built yet.
    """

    frame = pd.read_csv(SERUM_MANIFEST, sep="\t", dtype=str).fillna("")
    frame["include_in_hydrolysis_model"] = frame["include_in_hydrolysis_model"] == "true"
    return frame


def save_result(frame: pd.DataFrame, name: str) -> Path:
    """Write one intermediate result table to the analysis results directory.

    :param frame: Table to persist.
    :param name: File name including the ``.csv`` suffix.
    :return: Path of the written file.
    """

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_ROOT / name
    frame.to_csv(destination, index=False)
    return destination


def describe_source(path: Path) -> dict[str, Any]:
    """Summarize the availability of one acquisition artifact.

    Missing sources must stay visible as missing rather than collapsing into
    zeros, so every notebook reports availability through this helper.

    :param path: File or directory produced by the acquisition step.
    :return: Availability record with ``path``, ``exists`` and ``bytes``.
    """

    if path.is_dir():
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    elif path.is_file():
        size = path.stat().st_size
    else:
        size = 0
    return {"path": str(path.relative_to(ROOT)), "exists": path.exists(), "bytes": size}
