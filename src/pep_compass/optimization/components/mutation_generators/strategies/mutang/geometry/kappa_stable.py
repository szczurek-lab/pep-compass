"""Standalone MUTANG geometry computed explicitly from an autoencoder."""

from pep_compass.autoencoder.geometry import compute_stable_tangent_geometry


class KappaStableGeometry:
    """Compute independent point geometries for standalone MUTANG execution."""

    local_enumeration_only = False

    def __init__(self, *, kappa: float = 1e-6) -> None:
        self.kappa = kappa

    def resolve(self, batch, autoencoder):
        geometry = compute_stable_tangent_geometry(
            autoencoder, batch.latent_origins, kappa=self.kappa
        )
        return tuple(geometry.select(index) for index in range(len(batch)))
