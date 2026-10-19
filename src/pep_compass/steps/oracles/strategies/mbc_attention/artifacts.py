"""Artifact requirements owned by this model implementation."""

from pathlib import Path

from pep_compass.optimization.saving.artifacts import (
    DirectoryArtifact,
    ModelArtifacts,
)

ARTIFACTS = ModelArtifacts(
    provider="oracle.mbc_attention",
    name="default",
    directory=".",
    artifacts=(
        DirectoryArtifact(
            "model",
            required_files=(
                "saved_model.pb",
                "variables/variables.index",
                "variables/variables.data-00000-of-00001",
            ),
        ),
    ),
    base_directory=Path(__file__).parent,
)
