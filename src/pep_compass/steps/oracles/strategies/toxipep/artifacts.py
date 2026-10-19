"""Artifact requirements owned by this model implementation."""

from pathlib import Path

from pep_compass.optimization.saving.artifacts import (
    FileArtifact,
    ModelArtifacts,
)

ARTIFACTS = ModelArtifacts(
    provider="oracle.toxipep",
    name="default",
    directory="models",
    artifacts=(FileArtifact("best_model_0.9.pth"),),
    base_directory=Path(__file__).parent,
)
