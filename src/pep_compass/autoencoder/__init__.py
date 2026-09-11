"""Autoencoder contracts, implementations, model registry, and geometry."""

from pep_compass.autoencoder.base import Autoencoder
from pep_compass.autoencoder.factory import AutoencoderFactory
from pep_compass.autoencoder.geometry import TangentDecomposition

__all__ = ["Autoencoder", "AutoencoderFactory", "TangentDecomposition"]
