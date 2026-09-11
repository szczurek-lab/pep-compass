"""Fixed SORBES algorithm assembled from replaceable numerical stages."""

import torch

from pep_compass.optimization.components.walkers.strategies.sorbes.schema import SorbesStepResult


class Sorbes:
    """Advance points in the only supported SORBES stage order."""

    def __init__(self, autoencoder, geometry, directions, scaling, position_update, boundary):
        self.autoencoder = autoencoder
        self.geometry = geometry
        self.directions = directions
        self.scaling = scaling
        self.position_update = position_update
        self.boundary = boundary

    def step(self, positions, current_geometry=None):
        """Move points and return geometry belonging to the resulting points."""
        # Input-point geometry
        geometry = current_geometry or self.geometry(positions)

        # Direction selection and scaling
        directions = self.scaling(self.directions(geometry), geometry)

        # Manifold acceleration
        with torch.no_grad():
            ambient = self.autoencoder.field_derivative(
                positions, directions.active
            )  # (B, A)
        acceleration = geometry.project_ambient_to_active_latent(ambient)  # (B, D)

        # Position update and main boundary behaviour
        proposed, time_steps = self.position_update(
            positions, directions, acceleration
        )
        accepted = self.boundary(positions, proposed, geometry)  # (B, D)

        # Output-point geometry shared by MUTANG and the next SORBES step
        output_geometry = self.geometry(accepted)
        return SorbesStepResult(accepted, time_steps, output_geometry)
