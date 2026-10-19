"""Structural contract for deferred, shared autoencoder construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pep_compass.autoencoder.base import Autoencoder


@runtime_checkable
class AutoencoderResource(Protocol):
    """Describe an importable resource whose weights are loaded during build."""

    def to_config(self) -> Mapping[str, object]:
        """Return a portable constructor declaration.

        :return: Constructor target and resolved parameters.
        :rtype: Mapping[str, object]
        """
        ...

    def build(self, *, device: str | None = None) -> Autoencoder:
        """Create one numerical autoencoder shared by the compiled graph.

        :param device: Optional execution device overriding resource defaults.
        :type device: str | None
        :return: Initialized model implementing the numerical contract.
        :rtype: Autoencoder
        """
        ...
