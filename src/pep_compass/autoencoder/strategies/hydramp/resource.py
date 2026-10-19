"""Lightweight declaration of one shared HydrAMP model resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pep_compass.utils.configuration import Configurable
from pep_compass.utils.validation import validate_positive

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    import torch

    from pep_compass.autoencoder.base import Autoencoder


class HydrAMP(Configurable):
    """Configure the shared encoder; load weights only during build.

    :param model: Model subdirectory containing encoder and decoder weights.
    :param device: PyTorch computation device.
    :param jacobian_mode: Exact autograd or finite-difference Jacobians.
    :param jacobian_eps: Positive finite-difference increment.
    :param field_eps: Positive derivative increment for decoder fields.
    :param temp: Positive decoder temperature.
    :param default_condition: Two decoder conditioning values.
    """

    def __init__(
        self,
        *,
        model: str = "article_25",
        device: str | torch.device = "cpu",
        jacobian_mode: Literal["strict", "approx"] = "strict",
        jacobian_eps: float = 0.001,
        field_eps: float = 0.001,
        temp: float = 1.0,
        default_condition: Sequence[float] | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param model: Named model variant.
        :type model: str
        :param device: Computation device accepted by the numerical backend.
        :type device: str | torch.device
        :param jacobian_mode: Jacobian mode.
        :type jacobian_mode: Literal['strict', 'approx']
        :param jacobian_eps: Jacobian eps.
        :type jacobian_eps: float
        :param field_eps: Field eps.
        :type field_eps: float
        :param temp: Temp.
        :type temp: float
        :param default_condition: Default condition.
        :type default_condition: Sequence[float] | None
        :rtype: None
        """
        validate_positive(jacobian_eps, "HydrAMP.jacobian_eps")
        validate_positive(field_eps, "HydrAMP.field_eps")
        validate_positive(temp, "HydrAMP.temp")
        if jacobian_mode not in {"strict", "approx"}:
            raise ValueError("Invalid Jacobian mode.")
        condition = [1.0, 1.0] if default_condition is None else list(default_condition)
        if len(condition) != 2:
            raise ValueError("HydrAMP requires two conditioning values.")
        self._configure(
            dict(
                model=model,
                device=str(device),
                jacobian_mode=jacobian_mode,
                jacobian_eps=jacobian_eps,
                field_eps=field_eps,
                temp=temp,
                default_condition=condition,
            )
        )

    def build(self, *, device: str | None = None) -> Autoencoder:
        """Initialize the numerical model using this immutable constructor snapshot.

        :param device: Computation device accepted by the numerical backend.
        :type device: str | None
        :return: Result of this operation.
        :rtype: Autoencoder
        """
        import torch

        from pep_compass.autoencoder.strategies.hydramp.adapter import (
            HydrAMPModel,
        )

        values = self.configuration
        values["model_name"] = values.pop("model")
        values["device"] = device or values["device"]
        values["default_condition"] = torch.tensor(
            values["default_condition"], device=values["device"]
        )
        return HydrAMPModel(**values)
