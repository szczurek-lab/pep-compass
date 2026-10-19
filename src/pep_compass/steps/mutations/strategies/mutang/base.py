"""Typed stage interfaces for the fixed MUTANG algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

from pep_compass.autoencoder.geometry import StableTangentGeometry
from pep_compass.data_schemas.candidates import CandidateBatch
from pep_compass.steps.mutations.strategies.mutang.schema import (
    MutationOption,
    SelectedDirections,
)
from pep_compass.utils.configuration import Configurable

if TYPE_CHECKING:
    from pep_compass.autoencoder.base import Autoencoder


class GeometryStrategy(Configurable, ABC):
    """Resolve geometry aligned with the incoming candidates."""

    local_enumeration_only = False
    requirement = None

    @abstractmethod
    def resolve(
        self, batch: CandidateBatch, autoencoder: Autoencoder | None
    ) -> tuple[StableTangentGeometry, ...]:
        """Return one geometry row per candidate.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder | None
        :return: Result of this operation.
        :rtype: tuple[StableTangentGeometry, ...]
        """


class DirectionStrategy(Configurable, ABC):
    """Select singular directions for mutation scoring."""

    @abstractmethod
    def select(self, singular_values: torch.Tensor) -> SelectedDirections:
        """Return indices into the singular-vector axis.

        :param singular_values: Singular values.
        :type singular_values: torch.Tensor
        :return: Result of this operation.
        :rtype: SelectedDirections
        """


class ScoringStrategy(Configurable, ABC):
    """Score residue substitutions from selected tangent directions."""

    @abstractmethod
    def score(
        self, left_vectors: torch.Tensor, selected: SelectedDirections
    ) -> torch.Tensor:
        """Return substitution scores with shape (L, A).

        :param left_vectors: Left vectors.
        :type left_vectors: torch.Tensor
        :param selected: Selected.
        :type selected: SelectedDirections
        :return: Result of this operation.
        :rtype: torch.Tensor
        """


class MutationSelectionStrategy(Configurable, ABC):
    """Select scored substitutions for a sequence."""

    @abstractmethod
    def select(
        self, scores: torch.Tensor, sequence_length: int
    ) -> tuple[MutationOption, ...]:
        """Return admissible residue substitutions.

        :param scores: Scalar scores or a score tensor aligned with the corresponding candidates.
        :type scores: torch.Tensor
        :param sequence_length: Sequence length.
        :type sequence_length: int
        :return: Result of this operation.
        :rtype: tuple[MutationOption, ...]
        """


class CombinationStrategy(Configurable, ABC):
    """Enumerate candidate sequences from selected substitutions."""

    @abstractmethod
    def generate(
        self,
        sequence: str,
        options: tuple[MutationOption, ...],
        rng: np.random.Generator,
    ) -> Iterator[str]:
        """Yield candidates using the run-owned random generator.

        :param sequence: One peptide sequence.
        :type sequence: str
        :param options: Options.
        :type options: tuple[MutationOption, ...]
        :param rng: Run-owned random generator used for reproducible choices.
        :type rng: np.random.Generator
        :return: Result of this operation.
        :rtype: Iterator[str]
        """
