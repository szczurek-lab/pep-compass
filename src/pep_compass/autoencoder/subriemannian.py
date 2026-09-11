"""Shared sub-Riemannian geometry contracts derived from an autoencoder."""

from dataclasses import dataclass
from enum import Enum

import torch

from pep_compass.autoencoder.geometry import StableTangentGeometry


class GeometryArtifact(str, Enum):
    """Tensor products made available by a geometry provider."""

    LEFT_VECTORS = "left_vectors"
    SINGULAR_VALUES = "singular_values"
    RIGHT_VECTORS = "right_vectors"


@dataclass(frozen=True, slots=True)
class GeometryContract:
    """Describe geometry artifacts without tying them to a strategy name."""

    artifacts: frozenset[GeometryArtifact]
    source: str = "decoder_jacobian"

    def satisfies(self, requirement: "GeometryContract") -> bool:
        """Return whether this producer provides every required artifact."""
        return self.source == requirement.source and self.artifacts >= requirement.artifacts


TANGENT_GEOMETRY_CONTRACT = GeometryContract(frozenset(GeometryArtifact))
MUTANG_GEOMETRY_REQUIREMENT = GeometryContract(
    frozenset({GeometryArtifact.LEFT_VECTORS, GeometryArtifact.SINGULAR_VALUES})
)


@dataclass(frozen=True, slots=True)
class PointGeometry:
    """Bind one geometry row to the latent point for which it was computed."""

    point_id: int
    geometry: StableTangentGeometry
    provider_id: int

    def validate(self) -> None:
        """Validate the one-row structural invariant."""
        if self.geometry.singular_values.shape[0] != 1:
            raise ValueError("Point geometry must contain exactly one SVD row.")


class SubRiemannianTangentSpace:
    """Expose one geometry row to existing geometry-dependent filters."""

    def __init__(self, U, S, V, horizontal_threshold, device="cpu"):
        del device
        self.U, self.S, self.V = U, S, V
        self.horizontal_threshold, self.device = horizontal_threshold, U.device
        mask = torch.abs(S) > horizontal_threshold
        self.horizontal_dim, self.vertical_dim = int(mask.sum()), int((~mask).sum())
        self.projection_matrix = None

    def project_ambient_vector_to_horizontal_space(self, ambient_vector):
        """Apply the truncated pseudoinverse without recomputing an SVD."""
        if self.projection_matrix is None:
            inverse = torch.where(
                torch.abs(self.S) > self.horizontal_threshold,
                self.S.reciprocal(),
                torch.zeros_like(self.S),
            )
            self.projection_matrix = (self.V.T * inverse.unsqueeze(0)) @ self.U.T
        return self.projection_matrix @ ambient_vector
