"""Artifact requirements owned by this model implementation."""

from pathlib import Path

from pep_compass.optimization.saving.artifacts import (
    FileArtifact,
    ModelArtifacts,
)

ARTIFACTS = ModelArtifacts(
    provider="oracle.eipred",
    name="default",
    directory="DATA",
    artifacts=(
        FileArtifact("model2.pkl/model2.pkl"),
        FileArtifact("selected_features_mrmr1000_new.csv"),
    ),
    base_directory=Path(__file__).parent,
)
