"""Base contract for reusable candidate scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from pep_compass.utils.configuration import Configurable

if TYPE_CHECKING:
    from pep_compass.data_schemas.candidates import CandidateBatch
    from pep_compass.data_schemas.contracts import StepContract
    from pep_compass.optimization.engine.execution.context import OptimizationContext


class ScoreFunction(Configurable, ABC):
    """Assign one scalar score to every candidate in a batch."""

    higher_is_better: bool = True

    def contract(self) -> StepContract:
        """Declare optional resources and fields required by this scoring implementation.

        :return: Result of this operation.
        :rtype: StepContract
        """
        from pep_compass.data_schemas.contracts import StepContract

        return StepContract(
            resources=("autoencoder",) if hasattr(self, "autoencoder") else ()
        )

    @abstractmethod
    def __call__(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> torch.Tensor:
        """Return scores with shape ``(B,)``.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
