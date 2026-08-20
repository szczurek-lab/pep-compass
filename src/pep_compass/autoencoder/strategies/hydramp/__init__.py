"""HydrAMP autoencoder method and named-model registrations."""

from __future__ import annotations

from typing import Any
from threading import Lock

import torch

from pep_compass.autoencoder.registry import (
    AutoencoderModelDescriptor,
    AutoencoderRegistry,
)
from pep_compass.utils.strategy_factory import parameter_contract
from pep_compass.registry import FileArtifact


_registration_lock = Lock()
_registered = False


@parameter_contract(
    accepted={
        "device",
        "jacobian_mode",
        "default_condition",
        "temp",
        "jacobian_eps",
        "field_eps",
        "model_name",
    },
    required={"jacobian_eps", "field_eps"},
)
def build_hydramp(*, device: str = "cpu", **parameters: Any):
    """Build HydrAMP and normalize its configured condition tensor."""
    from pep_compass.autoencoder.strategies.hydramp.adapter import (
        HydrampAutoencoder,
    )

    if "default_condition" in parameters:
        parameters["default_condition"] = torch.as_tensor(
            parameters["default_condition"],
            device=device,
        )
    return HydrampAutoencoder(device=device, **parameters)


def register_hydramp() -> None:
    """Register the HydrAMP adapter and bundled named model variants."""
    global _registered
    if _registered:
        return
    with _registration_lock:
        if _registered:
            return
        AutoencoderRegistry.register_method("hydramp")(build_hydramp)
        AutoencoderRegistry.register_model(
            "hydramp",
            AutoencoderModelDescriptor(
                "article_25",
                {"model_name": "article_25"},
                directory="article_25",
                artifacts=(
                    FileArtifact("encoder_weights.pickle"),
                    FileArtifact("decoder_weights.pickle"),
                ),
            ),
        )
        _registered = True


__all__ = ["build_hydramp", "register_hydramp"]
