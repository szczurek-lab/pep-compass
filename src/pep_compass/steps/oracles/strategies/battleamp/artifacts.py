"""Artifact requirements owned by this model implementation."""

from pathlib import Path

from pep_compass.optimization.saving.artifacts import (
    FileArtifact,
    ModelArtifacts,
)

ARTIFACTS = ModelArtifacts(
    provider="oracle.battleamp",
    name="default",
    directory=".",
    artifacts=(FileArtifact("model.h5"),),
    base_directory=Path(__file__).parent,
)
