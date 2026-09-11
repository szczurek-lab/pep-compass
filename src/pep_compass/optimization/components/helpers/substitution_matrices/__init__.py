"""Substitution-matrix loading and sequence scoring."""

from pep_compass.optimization.components.helpers.substitution_matrices.scoring import (
    blosum_score,
    load_blosum,
)

__all__ = ["blosum_score", "load_blosum"]
