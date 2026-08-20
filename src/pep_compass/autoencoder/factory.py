"""Construction facade for registered autoencoder methods and models."""

from __future__ import annotations

from typing import Any

from pep_compass.autoencoder.base import Autoencoder
from pep_compass.autoencoder.registry import AutoencoderRegistry
from pep_compass.registry import load_builtin_registrations


class AutoencoderFactory:
    """Build autoencoders from an implementation method and named model."""

    @staticmethod
    def build(
        method: str,
        *,
        model: str,
        device: str = "cpu",
        **parameters: Any,
    ) -> Autoencoder:
        """Construct one configured autoencoder.

        :param method: Registered implementation method.
        :param model: Named model descriptor registered for the method.
        :param device: Target computation device.
        :param parameters: Explicit overrides for descriptor parameters.
        :return: Initialized autoencoder implementation.
        """
        load_builtin_registrations()

        descriptor = AutoencoderRegistry.model(method, model)
        resolved = {**descriptor.parameters, **parameters}
        return AutoencoderRegistry.method(method)(device=device, **resolved)
