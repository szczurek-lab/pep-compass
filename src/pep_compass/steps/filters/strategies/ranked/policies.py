"""Convenient compositions of real scoring and selection objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Ranked
from .scoring.latent_geometry.lams import LamsScore
from .scoring.latent_geometry.move import MoveScore
from .scoring.latent_geometry.tandem import TandemScore
from .scoring.model_scores.decoder_likelihood import DecoderLikelihoodScore
from .scoring.model_scores.esm import ESM2PPLScorer
from .selection.nucleus import NucleusSelection
from .selection.threshold import ThresholdSelection
from .selection.top_k import TopKSelection

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


def Lams(
    *,
    similarity_threshold: float = 0.15,
    horizontal_threshold: float = 0.1,
    maximum_candidates: int = 30000,
    alphabet: Sequence[str] | None = None,
) -> Ranked:
    """Compose LAMS scoring and threshold selection without a marker subclass.

    :param similarity_threshold: Similarity threshold.
    :type similarity_threshold: float
    :param horizontal_threshold: Horizontal threshold.
    :type horizontal_threshold: float
    :param maximum_candidates: Maximum materialized candidates allowed by this local strategy.
    :type maximum_candidates: int
    :param alphabet: Ordered residue alphabet.
    :type alphabet: Sequence[str] | None
    :return: Result of this operation.
    :rtype: Ranked
    """
    return Ranked(
        LamsScore(
            horizontal_threshold=horizontal_threshold,
            maximum_candidates=maximum_candidates,
            alphabet=alphabet,
        ),
        ThresholdSelection(similarity_threshold),
    )


def Tandem(
    *,
    top_p: float = 0.9,
    temperature: float = 1.0,
    horizontal_threshold: float = 0.1,
    maximum_candidates: int = 30000,
    alphabet: Sequence[str] | None = None,
) -> Ranked:
    """Compose TANDEM scoring with nucleus selection.

    :param top_p: Top p.
    :type top_p: float
    :param temperature: Temperature.
    :type temperature: float
    :param horizontal_threshold: Horizontal threshold.
    :type horizontal_threshold: float
    :param maximum_candidates: Maximum materialized candidates allowed by this local strategy.
    :type maximum_candidates: int
    :param alphabet: Ordered residue alphabet.
    :type alphabet: Sequence[str] | None
    :return: Result of this operation.
    :rtype: Ranked
    """
    return Ranked(
        TandemScore(
            horizontal_threshold=horizontal_threshold,
            maximum_candidates=maximum_candidates,
            alphabet=alphabet,
        ),
        NucleusSelection(top_p, temperature),
    )


def Move(
    *,
    top_p: float = 0.6,
    temperature: float = 1.0,
    maximum_candidates: int = 30000,
    alphabet: Sequence[str] | None = None,
) -> Ranked:
    """Compose MOVE scoring with nucleus selection.

    :param top_p: Top p.
    :type top_p: float
    :param temperature: Temperature.
    :type temperature: float
    :param maximum_candidates: Maximum materialized candidates allowed by this local strategy.
    :type maximum_candidates: int
    :param alphabet: Ordered residue alphabet.
    :type alphabet: Sequence[str] | None
    :return: Result of this operation.
    :rtype: Ranked
    """
    return Ranked(
        MoveScore(maximum_candidates=maximum_candidates, alphabet=alphabet),
        NucleusSelection(top_p, temperature),
    )


def LpBebo(
    *,
    top_p: float = 0.9,
    temperature: float = 1.0,
    maximum_candidates: int = 30000,
    alphabet: Sequence[str] | None = None,
) -> Ranked:
    """Compose decoder likelihood with nucleus selection.

    :param top_p: Top p.
    :type top_p: float
    :param temperature: Temperature.
    :type temperature: float
    :param maximum_candidates: Maximum materialized candidates allowed by this local strategy.
    :type maximum_candidates: int
    :param alphabet: Ordered residue alphabet.
    :type alphabet: Sequence[str] | None
    :return: Result of this operation.
    :rtype: Ranked
    """
    return Ranked(
        DecoderLikelihoodScore(
            maximum_candidates=maximum_candidates, alphabet=alphabet
        ),
        NucleusSelection(top_p, temperature),
    )


def ESM(
    *,
    threshold: float | None = None,
    top_k: int | None = None,
    model_name: str = "esm2_t6_8M_UR50D",
    device: str | torch.device = "cpu",
) -> Ranked:
    """Compose lazy ESM plausibility scoring with one explicit selection policy.

    :param threshold: Threshold.
    :type threshold: float | None
    :param top_k: Top k.
    :type top_k: int | None
    :param model_name: Named model variant.
    :type model_name: str
    :param device: Computation device accepted by the numerical backend.
    :type device: str | torch.device
    :return: Result of this operation.
    :rtype: Ranked
    """
    if (threshold is None) == (top_k is None):
        raise ValueError("Select threshold or top_k.")
    return Ranked(
        ESM2PPLScorer(model_name, device),
        ThresholdSelection(threshold) if top_k is None else TopKSelection(top_k),
    )
