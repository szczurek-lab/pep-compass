"""Proteolysis environment shared across optimization components.

This subpackage holds the reusable *environment* for modelling proteolytic
(exo- and endopeptidase) cleavage: loading MEROPS protease specificity panels
and evaluating the per-bond cleavage susceptibility :math:`\\Phi_{cleav}`. It is
deliberately placed under ``components/helpers`` rather than under a specific
filter, because the same physics is consumed by several components (ranked
CLASP scoring, oracles/black boxes, and analysis notebooks).
"""

from pep_compass.optimization.components.helpers.proteolysis.cleavage import (
    DEFAULT_MEROPS_DATASETS,
    MEROPS_AMINO_ACIDS,
    MEROPS_SUBSITE_OFFSETS,
    MEROPS_SUBSITES,
    CleavagePotential,
    ProteasePanel,
    load_protease_panel,
)

__all__ = [
    "DEFAULT_MEROPS_DATASETS",
    "MEROPS_AMINO_ACIDS",
    "MEROPS_SUBSITE_OFFSETS",
    "MEROPS_SUBSITES",
    "CleavagePotential",
    "ProteasePanel",
    "load_protease_panel",
]
