r"""CLASP ranked-scoring objective combining APEX activity and cleavage stability.

This module holds the *ranking objective* used to score candidate mutations,
as opposed to the proteolysis *environment* (protease panels and the cleavage
physics :math:`\Phi_{cleav}`), which lives in
``pep_compass.optimization.components.helpers.proteolysis``.

``ClaspPotential`` is a :class:`MutationPotential`: it ranks complete MUTANG
mutation combinations by ``-(log2 MIC + lambda * Phi_cleav)`` so that higher is
both more active (lower MIC) and more stable (lower cleavage). ``build_clasp_potential``
and ``compose_clasp_distribution`` are convenience constructors that assemble a
:class:`ClaspPotential` from MEROPS datasets and APEX predictors.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch

from pep_compass.optimization.components.filters.ranked.scoring.helpers.mutation_potentials import (  # noqa: E501
    DEFAULT_MAX_LEN,
    MutantDistribution,
    MutationPotential,
    compose_mutant_distribution,
)
from pep_compass.optimization.components.helpers.proteolysis.cleavage import (
    DEFAULT_MEROPS_DATASETS,
    CleavagePotential,
    load_protease_panel,
)

# Default APEX pathogen column remains S. aureus ATCC 12600 (id 7).
DEFAULT_APEX_INDEX = 7


class ClaspPotential(MutationPotential):
    r"""CLASP objective combining APEX activity and cleavage stability.

    Implements the brief's target :math:`\Phi = \log_2 \mathrm{MIC} + \lambda\,
    \Phi_{cleav}` (the brief calls the trade-off weight :math:`\gamma`). Because
    the enumeration framework ranks candidates by descending score, this class
    returns the negated objective ``-(log2 MIC + lambda * Phi_cleav)`` so that
    higher means both more active (lower MIC) and more stable (lower cleavage).

    APEX is used two ways: :meth:`score_sequences` / :meth:`compute` rank
    concrete MUTANG candidates through the string predictor, and
    :meth:`distribution_potential` provides a differentiable oracle over decoder
    outputs for PoGS/LE-BO.
    """

    def __init__(
        self,
        cleavage_potential: CleavagePotential,
        apex_predictor=None,
        apex_probs_predictor=None,
        lam: float = 0.1,
        apex_index: int = DEFAULT_APEX_INDEX,
        alphabet: list[str] | None = None,
    ):
        """Initialize the combined CLASP potential.

        :param cleavage_potential: Cleavage term :math:`\\Phi_{cleav}`.
        :param apex_predictor: ``PredictorAPEX`` (string input) for ranking.
        :param apex_probs_predictor: ``PredictorAPEX_Probs`` (decoder-output
            input) for the differentiable oracle.
        :param lam: Activity/stability trade-off weight :math:`\\lambda`.
        :param apex_index: APEX pathogen column used for ``log2 MIC``. Defaults
            to S. aureus ATCC 12600 (column 7).
        :param alphabet: Optional index-to-token mapping.
        """
        self.cleavage = cleavage_potential
        self.apex_predictor = apex_predictor
        self.apex_probs_predictor = apex_probs_predictor
        self.lam = float(lam)
        self.apex_index = apex_index
        self.alphabet = alphabet or cleavage_potential.alphabet

    def score_components(self, sequences: Sequence[str]) -> dict[str, np.ndarray]:
        r"""Return the individual CLASP terms for concrete peptides.

        The activity and stability contributions are reported separately so
        they can be inspected, compared, or plotted against each other instead
        of only through their weighted sum.

        :param sequences: Peptide strings.
        :return: Arrays keyed ``"log2_mic"`` (APEX activity term),
            ``"cleavage"`` (:math:`\Phi_{cleav}`), ``"cleavage_term"``
            (:math:`\lambda \Phi_{cleav}`) and ``"score"`` (the value returned
            by :meth:`score_sequences`).
        :raises ValueError: If no string APEX predictor was provided.
        """
        if self.apex_predictor is None:
            raise ValueError("score_components requires apex_predictor")
        keys = ("log2_mic", "cleavage", "cleavage_term", "score")
        if len(sequences) == 0:
            return {key: np.empty(0, dtype=np.float64) for key in keys}
        sequences = list(sequences)
        mic = np.asarray(self.apex_predictor.predict(sequences))
        log2_mic = np.log2(mic[:, self.apex_index])
        cleavage = self.cleavage.sequence_log_potential(sequences).cpu().numpy()
        cleavage_term = self.lam * cleavage
        return {
            "log2_mic": log2_mic,
            "cleavage": cleavage,
            "cleavage_term": cleavage_term,
            "score": -(log2_mic + cleavage_term),
        }

    def score_sequences(self, sequences: Sequence[str]) -> np.ndarray:
        """Return the negated CLASP objective for concrete peptides.

        :param sequences: Peptide strings.
        :return: ``-(log2 MIC + lambda * Phi_cleav)`` per sequence.
        :raises ValueError: If no string APEX predictor was provided.
        """
        return self.score_components(sequences)["score"]

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[tuple[int, ...], float]:
        r"""Score every non-parent combination with the CLASP objective.

        :param parent_peptide: Parent peptide sequence.
        :param mutations: Candidate amino-acid indices grouped by position.
        :return: Complete amino-acid tuples (ordered by ``sorted(mutations)``)
            mapped to ``-(log2 MIC + lambda * Phi_cleav)``.
        """
        positions = sorted(mutations)
        if not positions:
            return {}
        padded_parent = parent_peptide.ljust(DEFAULT_MAX_LEN)
        length = len(parent_peptide)
        choices = [mutations[position] for position in positions]

        tuples: list[tuple[int, ...]] = []
        sequences: list[str] = []
        for combination in itertools.product(*choices):
            residue = list(padded_parent)
            for position, amino_acid in zip(positions, combination):
                residue[position] = self.alphabet[amino_acid]
            candidate = "".join(residue[:length])
            if candidate == parent_peptide:
                continue
            tuples.append(tuple(combination))
            sequences.append(candidate)

        if not sequences:
            return {}
        scores = self.score_sequences(sequences)
        return {key: float(score) for key, score in zip(tuples, scores)}

    def distribution_potential(self, decoder_output: torch.Tensor) -> torch.Tensor:
        r"""Return the differentiable CLASP objective for decoder outputs.

        :param decoder_output: Decoder residue distribution ``Dec(z)`` with a
            trailing vocabulary axis, shape ``(..., L, V)``.
        :return: ``-(log2 MIC + lambda * Phi_cleav)`` per sequence, shape
            ``(B,)``, differentiable with respect to ``decoder_output``.
        :raises ValueError: If no differentiable APEX predictor was provided.
        """
        if self.apex_probs_predictor is None:
            raise ValueError("distribution_potential requires apex_probs_predictor")
        mic = self.apex_probs_predictor.predict(decoder_output)
        log2_mic = torch.log2(mic[:, self.apex_index])
        cleavage = self.cleavage.distribution_log_potential(decoder_output)
        return -(log2_mic + self.lam * cleavage)


def build_clasp_potential(
    apex_predictor=None,
    apex_probs_predictor=None,
    dataset_ids: int | Iterable[int] = DEFAULT_MEROPS_DATASETS,
    merops_root: str | Path | None = None,
    variant: str = "product",
    lam: float = 0.1,
    apex_index: int = DEFAULT_APEX_INDEX,
    alphabet: list[str] | None = None,
    weights: Sequence[float] | None = None,
    background: torch.Tensor | Sequence[float] | None = None,
    temperature: float = 1.0,
    device: str | torch.device = "cpu",
) -> ClaspPotential:
    """Build a :class:`ClaspPotential` from MEROPS datasets and APEX predictors.

    :param apex_predictor: ``PredictorAPEX`` for ranking concrete candidates.
    :param apex_probs_predictor: ``PredictorAPEX_Probs`` for the differentiable
        oracle.
    :param dataset_ids: MEROPS dataset id(s) for the protease panel (default
        human serum, id 35).
    :param merops_root: Optional MEROPS directory override.
    :param variant: Cleavage variant (``"product"`` by default).
    :param lam: Activity/stability trade-off weight.
    :param apex_index: APEX pathogen column for ``log2 MIC``.
    :param alphabet: Optional index-to-token mapping.
    :param weights: Optional protease weights.
    :param background: Optional background residue distribution.
    :param temperature: Mean-field temperature.
    :param device: Device for the potential.
    :return: A configured :class:`ClaspPotential`.
    """
    panel = load_protease_panel(
        dataset_ids, merops_root=merops_root, weights=weights, device=device
    )
    cleavage = CleavagePotential(
        panel,
        variant=variant,
        alphabet=alphabet,
        background=background,
        temperature=temperature,
        device=device,
    )
    return ClaspPotential(
        cleavage,
        apex_predictor=apex_predictor,
        apex_probs_predictor=apex_probs_predictor,
        lam=lam,
        apex_index=apex_index,
        alphabet=alphabet,
    )


def compose_clasp_distribution(
    parent_peptide: str,
    mutations: dict[int, list[int]],
    apex_predictor=None,
    clasp_potential: ClaspPotential | None = None,
    alphabet: list[str] | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    maximum_candidates: int | None = None,
    **build_kwargs,
) -> MutantDistribution:
    """Compose and rank MUTANG candidates by the CLASP objective.

    A convenience wrapper that builds a :class:`ClaspPotential` (unless one is
    supplied) and delegates to :func:`compose_mutant_distribution`. Because the
    potential is tuple-keyed, candidates are scored as complete combinations.

    :param parent_peptide: Parent peptide sequence.
    :param mutations: Candidate amino-acid indices grouped by position.
    :param apex_predictor: ``PredictorAPEX`` used when building a potential.
    :param clasp_potential: Optional prebuilt potential; overrides construction.
    :param alphabet: Optional index-to-token mapping.
    :param max_len: Padding length for the parent peptide.
    :param maximum_candidates: Optional cap on the returned highest-scoring rows.
    :param build_kwargs: Forwarded to :func:`build_clasp_potential`.
    :return: Sorted mutant distribution under the CLASP objective.
    """
    if clasp_potential is None:
        clasp_potential = build_clasp_potential(
            apex_predictor=apex_predictor, alphabet=alphabet, **build_kwargs
        )
    return compose_mutant_distribution(
        parent_peptide,
        mutations,
        clasp_potential,
        alphabet=alphabet,
        max_len=max_len,
        maximum_candidates=maximum_candidates,
    )
