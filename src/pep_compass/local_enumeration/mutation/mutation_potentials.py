"""Potentials used to score combinations of MUTANG mutations.

This module consolidates reusable historical implementations without retaining
their obsolete import path or script-local copies:

* ``DecoderLogProbPotential``, ``ProjectedDirectionPairwiseSimilarityPotential``,
  ``AmbientMetricPairwiseSimilarityPotential``, and
  ``compose_mutant_distribution`` come from
  ``upstream/kjxpp/main:src/pep_compass/local_enumeration/mutation/``
  ``mutation_potentials.py`` and their extended ``upstream/rl_trials`` version;
* ``LamsAnchorSimilarityPotential`` replaces ``_MutangPlusProductPotential``
  embedded in ``upstream/rl_trials:scripts/lebo_plus.py``;
* ``SubRiemannianTangentSpace`` is imported from the canonical
  ``sampling_walker.py`` on ``dev`` instead of copying historical
  ``sampling/sorbes.py``.

Experiment orchestration belongs to filters and local enumerators. Potentials
only represent and score mutation choices, which lets the same implementation
be reused by runners instead of being duplicated in each experiment script.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import itertools
import json

import numpy as np
import torch

# NOTE: Nie przenosiłem `sampling/sorbes.py`; całość jest zaimplementowana w `pep_compass.local_enumeration.sampling_walker`.
from pep_compass.local_enumeration.sampling_walker import SubRiemannianTangentSpace
from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)

DEFAULT_ALPHABET = list(" ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MAX_LEN = 25

# MEROPS specificity-matrix layout (see data/merops/README.md and index.json).
# Amino-acid columns follow the APEX/MEROPS order, which is identical to
# ``DEFAULT_ALPHABET`` without the leading pad token.
MEROPS_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# Subsite (row) order of ``matrices_logprob.npy`` axis 1.
MEROPS_SUBSITES = ("P4", "P3", "P2", "P1", "P1p", "P2p", "P3p", "P4p")
# Position of each subsite relative to a bond cut between residues ``i`` and
# ``i + 1`` (the cut sits between P1 and P1'): P4 P3 P2 P1 | P1' P2' P3' P4'.
MEROPS_SUBSITE_OFFSETS = (-3, -2, -1, 0, 1, 2, 3, 4)
# Default protease panel: S. aureus ATCC 12600 (dataset id 7 == APEX pathogen
# column 7). Homo sapiens is id 34; other pathogens are ids 0-33.
DEFAULT_MEROPS_DATASETS = (7,)
DEFAULT_APEX_INDEX = 7


class MutantDistribution(NamedTuple):
    """Scored peptide candidates ordered from the highest potential."""

    sequences: list[str]
    log_potentials: np.ndarray  # 1-D float64, sorted descending


class MutationPotential(ABC):
    """Base class for mutation potential functions.

    Subclasses implement ``compute``, which maps a parent peptide and a set of
    candidate single-position mutations either to independent residue scores or
    to scores for complete Cartesian-product combinations.
    """

    @abstractmethod
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[int, dict[int, float]] | dict[tuple[int, ...], float]:
        """Return scores for candidate mutation choices.

        Args:
            parent_peptide: Parent peptide sequence.
            mutations: Position indices mapped to candidate amino-acid indices,
                in the format returned by ``get_mutations_from_s_u``.

        Returns:
            Either per-position scores ``{position: {aa_index: score}}`` or
            combination scores ``{aa_index_tuple: score}``. Tuple elements
            follow ``sorted(mutations)``.
        """


class DecoderLogProbabilityPotential(MutationPotential):
    """Log-probability of each mutant residue under the parent distribution."""

    def __init__(
        self,
        encoder_decoder: HydrAMPEncoderDecoder,
        alphabet: list[str] | None = None,
    ):
        """Initialize parent-conditioned decoder scoring.

        :param encoder_decoder: Model used to encode the parent and evaluate its
            decoder distribution.
        :param alphabet: Optional index-to-token mapping.
        """
        self.encoder_decoder = encoder_decoder
        self.alphabet = alphabet or DEFAULT_ALPHABET

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[int, dict[int, float]]:
        """Return decoder log-probability for every proposed residue.

        :param parent_peptide: Sequence defining the latent decoder condition.
        :param mutations: Candidate amino-acid indices grouped by position.
        :return: Nested position and amino-acid log-probabilities.
        """
        latent = self.encoder_decoder.encode_peptides([parent_peptide])
        log_probabilities = self.encoder_decoder.decoder_forward(
            latent,
            softmax=False,
            log_softmax=True,
            flatten=False,
        )[0]  # (max_len, alphabet_size)
        return {
            position: {
                amino_acid: log_probabilities[position, amino_acid].item()
                for amino_acid in amino_acids
            }
            for position, amino_acids in mutations.items()
        }


class ProjectedDirectionPairwiseSimilarityPotential(MutationPotential):
    r"""TANDEM potential based on pairwise projected mutation directions.

    For every complete combination, mutation directions are compared pairwise.
    Mutated-mutated pairs contribute ``log((1 + cos) / 2)`` and pairs in which
    exactly one residue is mutated contribute ``log((1 - cos) / 2)``. The
    contributions are averaged over all pairs involving at least one mutation.
    A single-position mutation has score zero because it has no pair.

    The returned key is a tuple of amino-acid indices ordered according to
    ``sorted(mutations)``. The all-parent combination is excluded.

    ``onehot`` represents a change at position :math:`l` to residue :math:`a`
    by :math:`e_{l,a}`. ``diff`` represents the actual displacement from the
    parent, :math:`e_{l,a} - e_{l,p_l}`.
    """

    def __init__(
        self,
        tangent_space: SubRiemannianTangentSpace,
        alphabet: list[str] | None = None,
        taken_taken_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        taken_not_taken_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        direction_mode: str = "onehot",
    ):
        """Initialize TANDEM projected-direction scoring.

        :param tangent_space: Local SVD geometry used to pull ambient residue
            directions into horizontal latent space.
        :param alphabet: Optional index-to-token mapping.
        :param taken_taken_transform: Score transform for pairs in which both
            residue choices mutate.
        :param taken_not_taken_transform: Score transform for pairs in which
            exactly one residue choice mutates.
        :param direction_mode: ``onehot`` for target directions or ``diff`` for
            target-minus-parent directions.
        :raises ValueError: If ``direction_mode`` is unsupported.
        """
        if direction_mode not in {"onehot", "diff"}:
            raise ValueError("direction_mode must be 'onehot' or 'diff'")
        self.tangent_space = tangent_space
        self.alphabet = alphabet or DEFAULT_ALPHABET
        # "onehot": the direction is the ambient one-hot e_(position, target).
        # "diff": it is e_(position, target) - e_(position, parent), which
        # represents movement from the current residue to the target residue.
        self.direction_mode = direction_mode
        self.taken_taken_transform = taken_taken_transform or self._log_taken_taken
        self.taken_not_taken_transform = (
            taken_not_taken_transform or self._log_taken_not_taken
        )

    @staticmethod
    def _log_taken_taken(values: torch.Tensor) -> torch.Tensor:
        """Map aligned jointly selected directions to higher log-potentials."""
        return torch.log(torch.clamp((1.0 + values) / 2.0, min=1e-12, max=1.0))

    @staticmethod
    def _log_taken_not_taken(values: torch.Tensor) -> torch.Tensor:
        """Penalize selecting only one member of an aligned direction pair."""
        return torch.log(torch.clamp((1.0 - values) / 2.0, min=1e-12, max=1.0))

    def _projection_matrix(self) -> torch.Tensor:
        """Return the cached horizontal ambient-to-latent projection matrix."""
        if self.tangent_space.projection_matrix is None:
            ambient_dimension = DEFAULT_MAX_LEN * len(self.alphabet)
            self.tangent_space.project_ambient_vector_to_horizontal_space(
                torch.zeros(
                    ambient_dimension,
                    device=self.tangent_space.device,
                    dtype=self.tangent_space.U.dtype,
                )
            )
        return self.tangent_space.projection_matrix

    def _raw_directions(self, flat_indices: torch.Tensor) -> torch.Tensor:
        r"""Represent ambient one-hot directions by latent pull-backs.

        This is Variant A from ``rl_trials``: ``J_h^+ e`` is read from columns
        of the horizontal projection matrix. Euclidean cosine between these
        vectors is therefore a whitened, inverse-singular-value-weighted
        similarity.
        """
        return self._projection_matrix()[:, flat_indices].T

    def position_vectors(
        self,
        position: int,
        amino_acids: torch.Tensor,
        parent_amino_acid: int,
    ) -> torch.Tensor:
        """Return normalized candidate directions for one sequence position.

        In ``onehot`` mode this returns representations of ``e_(position, aa)``.
        In ``diff`` mode the parent representation is subtracted before
        normalization; the identity choice consequently becomes a zero vector.
        """
        flat_indices = position * len(self.alphabet) + amino_acids
        vectors = self._raw_directions(flat_indices)
        if self.direction_mode == "diff":
            parent_index = torch.tensor(
                [position * len(self.alphabet) + parent_amino_acid],
                device=amino_acids.device,
            )
            vectors = vectors - self._raw_directions(parent_index)  # Broadcast (1, d).
        return vectors / (torch.linalg.norm(vectors, dim=1, keepdim=True) + 1e-12)

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[tuple[int, ...], float]:
        """Score every non-parent combination with the TANDEM pair rule.

        :param parent_peptide: Sequence defining identity residue choices.
        :param mutations: Candidate amino-acid indices grouped by position.
        :return: Complete amino-acid tuples mapped to mean pair potentials.
        """
        positions = sorted(mutations)
        if not positions:
            return {}

        # Parent amino-acid indices identify identity choices in the product.
        padded_parent = parent_peptide.ljust(DEFAULT_MAX_LEN)
        device = self.tangent_space.device
        parent_amino_acids = torch.tensor(
            [self.alphabet.index(padded_parent[position]) for position in positions],
            device=device,
            dtype=torch.long,
        )
        choices = [
            torch.tensor(mutations[position], device=device, dtype=torch.long)
            for position in positions
        ]
        vectors = [
            self.position_vectors(position, amino_acids, int(parent_amino_acids[i]))
            for i, (position, amino_acids) in enumerate(zip(positions, choices))
        ]
        ranges = [torch.arange(len(choice), device=device) for choice in choices]
        combination_indices = (
            ranges[0].unsqueeze(1)
            if len(ranges) == 1
            else torch.cartesian_prod(*ranges)
        )
        combinations = torch.stack(
            [choices[i][combination_indices[:, i]] for i in range(len(positions))],
            dim=1,
        )
        mutation_mask = combinations != parent_amino_acids
        valid = mutation_mask.any(dim=1)

        if len(positions) == 1:
            scores = torch.zeros(len(combinations), device=device)
        else:
            selected_vectors = torch.stack(
                [vectors[i][combination_indices[:, i]] for i in range(len(positions))],
                dim=1,
            )
            similarities = selected_vectors @ selected_vectors.transpose(1, 2)
            left, right = torch.triu_indices(
                len(positions), len(positions), offset=1, device=device
            )
            pair_similarities = similarities[:, left, right]
            left_mutated = mutation_mask[:, left]
            right_mutated = mutation_mask[:, right]
            both_mutated = left_mutated & right_mutated
            one_mutated = left_mutated ^ right_mutated
            included = both_mutated | one_mutated
            scores = (
                self.taken_taken_transform(pair_similarities) * both_mutated
                + self.taken_not_taken_transform(pair_similarities) * one_mutated
            ).sum(dim=1) / included.sum(dim=1).clamp(min=1)

        return {
            tuple(amino_acids): float(score)
            for amino_acids, score in zip(
                combinations[valid].cpu().tolist(), scores[valid].cpu().tolist()
            )
        }


class AmbientMetricPairwiseSimilarityPotential(
    ProjectedDirectionPairwiseSimilarityPotential
):
    r"""TANDEM variant B using the stable decoder pullback projector.

    Let :math:`U_\kappa` contain the decoder Jacobian's left singular vectors
    retained by the horizontal threshold. An ambient mutation direction
    :math:`d_i` is represented by :math:`U_\kappa^T d_i`, so the cosine used by
    TANDEM is equivalent to the normalized bilinear form induced by
    :math:`G_\kappa = U_\kappa U_\kappa^T`.

    Variant A, :class:`ProjectedDirectionPairwiseSimilarityPotential`, remains
    the recommended default because its inverse-singular-value whitening keeps
    useful latent-proximity information. Variant B is retained for thesis
    reproduction and comparisons of the two similarity definitions.
    """

    def __init__(
        self,
        tangent_space: SubRiemannianTangentSpace,
        alphabet: list[str] | None = None,
        taken_taken_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        taken_not_taken_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        direction_mode: str = "onehot",
    ):
        """Initialize TANDEM with stable ambient pullback geometry.

        :param tangent_space: Local decoder SVD defining the stable subspace.
        :param alphabet: Optional index-to-token mapping.
        :param taken_taken_transform: Score transform for two selected
            mutations. The logarithmic TANDEM transform is used by default.
        :param taken_not_taken_transform: Score transform for exactly one
            selected mutation. The logarithmic TANDEM transform is used by
            default.
        :param direction_mode: ``onehot`` for target directions or ``diff`` for
            target-minus-parent directions.
        :raises ValueError: If ``direction_mode`` is unsupported.
        """
        super().__init__(
            tangent_space=tangent_space,
            alphabet=alphabet,
            taken_taken_transform=taken_taken_transform,
            taken_not_taken_transform=taken_not_taken_transform,
            direction_mode=direction_mode,
        )
        horizontal = torch.abs(tangent_space.S) > tangent_space.horizontal_threshold
        self.horizontal_basis = tangent_space.U[:, horizontal].contiguous()

    def _raw_directions(self, flat_indices: torch.Tensor) -> torch.Tensor:
        r"""Return :math:`U_\kappa^T e_i` representations for ambient indices.

        :param flat_indices: Flattened position-residue indices.
        :return: Rows of the stable ambient basis corresponding to the supplied
            mutation directions.
        """
        return self.horizontal_basis[flat_indices]


class LamsAnchorSimilarityPotential(MutationPotential):
    """LAMS hard-viability score for a complete mutation combination.

    This is the product variant from ``rl_trials/scripts/lebo_plus.py``. For
    every pair of mutated residues, the whitened cosine is computed. The
    candidate score is the minimum across those pairs, so every mutation in a
    multi-mutant must remain compatible with every other mutation.
    A candidate with one mutation has score ``+inf`` and therefore always passes
    a finite downstream threshold. The all-parent candidate is excluded.
    """

    def __init__(self, base_potential: ProjectedDirectionPairwiseSimilarityPotential):
        """Initialize LAMS from a projected-direction representation.

        :param base_potential: Potential providing normalized position vectors
            and the associated tangent space.
        """
        self.base_potential = base_potential
        self.alphabet = base_potential.alphabet

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[tuple[int, ...], float]:
        """Return the worst mutated-mutated cosine for each combination.

        :param parent_peptide: Sequence defining identity residue choices.
        :param mutations: Candidate amino-acid indices grouped by position.
        :return: Non-parent combinations mapped to global minimum cosine; a
            single mutation receives positive infinity.
        """
        positions = sorted(mutations)
        if not positions:
            return {}
        device = self.base_potential.tangent_space.device
        padded_parent = parent_peptide.ljust(DEFAULT_MAX_LEN)
        parents = torch.tensor(
            [self.alphabet.index(padded_parent[position]) for position in positions],
            device=device,
            dtype=torch.long,
        )
        choices = [
            torch.tensor(mutations[position], device=device, dtype=torch.long)
            for position in positions
        ]
        vectors = [
            self.base_potential.position_vectors(position, choice, int(parents[i]))
            for i, (position, choice) in enumerate(zip(positions, choices))
        ]
        ranges = [torch.arange(len(choice), device=device) for choice in choices]
        indices = (
            ranges[0].unsqueeze(1)
            if len(ranges) == 1
            else torch.cartesian_prod(*ranges)
        )
        combinations = torch.stack(
            [choices[i][indices[:, i]] for i in range(len(positions))], dim=1
        )
        mutation_mask = combinations != parents
        valid = mutation_mask.any(dim=1)
        if len(positions) == 1:
            scores = torch.full((len(combinations),), float("inf"), device=device)
        else:
            selected = torch.stack(
                [vectors[i][indices[:, i]] for i in range(len(positions))], dim=1
            )
            similarities = selected @ selected.transpose(1, 2)
            left, right = torch.triu_indices(
                len(positions), len(positions), offset=1, device=device
            )
            pair_similarities = similarities[:, left, right]
            both_mutated = mutation_mask[:, left] & mutation_mask[:, right]
            masked_similarities = torch.where(
                both_mutated,
                pair_similarities,
                torch.full_like(pair_similarities, float("inf")),
            )
            has_pair = both_mutated.any(dim=1)
            scores = masked_similarities.min(dim=1).values
            scores = torch.where(
                has_pair, scores, torch.full_like(scores, float("inf"))
            )
        return {
            tuple(amino_acids): float(score)
            for amino_acids, score in zip(
                combinations[valid].cpu().tolist(), scores[valid].cpu().tolist()
            )
        }


def compose_mutant_distribution(
    parent_peptide: str,
    mutations: dict[int, list[int]],
    potential: MutationPotential,
    alphabet: list[str] | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    include_parent_residue: bool = False,
    maximum_candidates: int | None = None,
) -> MutantDistribution:
    """Compose, score, and sort the Cartesian product of mutation choices.

    Per-position potentials are added across positions. Tuple-keyed potentials
    already describe complete combinations and are materialized directly. When
    requested, the parent's residue is added at each position so the product
    includes candidates mutating only a subset of the available positions.
    ``maximum_candidates`` limits the returned highest-scoring rows.
    """
    alphabet = alphabet or DEFAULT_ALPHABET
    padded_parent = parent_peptide.ljust(max_len)
    augmented = {
        position: sorted(
            set(amino_acids)
            | (
                {alphabet.index(padded_parent[position])}
                if include_parent_residue
                else set()
            )
        )
        for position, amino_acids in mutations.items()
    }
    potentials = potential.compute(parent_peptide, augmented)
    if not potentials:
        return MutantDistribution([], np.array([], dtype=np.float64))

    positions = sorted(augmented)
    sequences: list[str] = []
    scores: list[float] = []
    if isinstance(next(iter(potentials)), tuple):
        for amino_acids, score in potentials.items():
            sequence = list(padded_parent)
            for position, amino_acid in zip(positions, amino_acids):
                sequence[position] = alphabet[amino_acid]
            sequences.append("".join(sequence[: len(parent_peptide)]))
            scores.append(score)
    else:
        choices = [list(potentials[position]) for position in positions]
        for amino_acids in itertools.product(*choices):
            sequence = list(padded_parent)
            score = 0.0
            for position, amino_acid in zip(positions, amino_acids):
                sequence[position] = alphabet[amino_acid]
                score += potentials[position][amino_acid]
            candidate = "".join(sequence[: len(parent_peptide)])
            if candidate != parent_peptide:
                sequences.append(candidate)
                scores.append(score)

    order = np.argsort(scores)[::-1]
    if maximum_candidates is not None:
        order = order[:maximum_candidates]
    return MutantDistribution(
        [sequences[index] for index in order],
        np.asarray([scores[index] for index in order], dtype=np.float64),
    )

class ProteasePanel(NamedTuple):
    """A set of proteases with specificity matrices and relative weights.

    ``matrices`` are the per-protease specificity matrices :math:`M_\\pi`
    (``matrices_logprob.npy`` from MEROPS by default), with axes
    ``[protease, subsite, amino_acid]``; ``weights`` are the panel weights
    :math:`\\rho_\\pi \\ge 0`; ``codes`` are the MEROPS peptidase codes, one per
    protease row.
    """

    matrices: torch.Tensor  # (N, 8, 20) float32
    weights: torch.Tensor  # (N,) float32
    codes: list[str]


def _default_merops_root() -> Path:
    """Return the repository ``data/merops`` directory.

    The module lives at ``src/pep_compass/local_enumeration/mutation`` so the
    repository root is four parents up.
    """
    return Path(__file__).resolve().parents[4] / "data" / "merops"


def load_protease_panel(
    dataset_ids: int | Iterable[int] = DEFAULT_MEROPS_DATASETS,
    merops_root: str | Path | None = None,
    kind: str = "logprob",
    weights: Sequence[float] | None = None,
    device: str | torch.device = "cpu",
) -> ProteasePanel:
    """Load and concatenate MEROPS specificity matrices for the given datasets.

    :param dataset_ids: Integer dataset id or ids. Ids 0-33 are the APEX
        pathogen columns (in ``PredictorAPEX(path="all").pathogen_list`` order),
        and id 34 is Homo sapiens. Defaults to S. aureus ATCC 12600 (id 7).
    :param merops_root: Directory containing ``index.json`` and the per-dataset
        folders. Defaults to the repository ``data/merops``.
    :param kind: ``"logprob"`` (specificity matrix :math:`M_\\pi`, the default)
        or ``"counts"`` (raw cleavage counts).
    :param weights: Optional protease weights :math:`\\rho_\\pi`, one per row of
        the concatenated panel. Defaults to uniform ones.
    :param device: Device for the returned tensors.
    :return: A :class:`ProteasePanel` spanning all requested datasets.
    """
    if isinstance(dataset_ids, int):
        dataset_ids = [dataset_ids]
    else:
        dataset_ids = list(dataset_ids)
    root = Path(merops_root) if merops_root is not None else _default_merops_root()
    index = json.loads((root / "index.json").read_text())
    by_index = index["by_index"]

    matrices: list[np.ndarray] = []
    codes: list[str] = []
    for dataset_id in dataset_ids:
        entry = by_index[str(dataset_id)]
        folder = root / entry["folder"]
        matrix = np.load(folder / f"matrices_{kind}.npy").astype(np.float32)
        metadata = json.loads((folder / "codes.json").read_text())
        peptidases = sorted(metadata["peptidases"], key=lambda item: item["row"])
        if matrix.shape[0] != len(peptidases):
            raise ValueError(
                f"Dataset {dataset_id} has {matrix.shape[0]} matrices but "
                f"{len(peptidases)} peptidase entries in codes.json"
            )
        matrices.append(matrix)
        codes.extend(peptidase["code"] for peptidase in peptidases)

    stacked = np.concatenate(matrices, axis=0)
    matrix_tensor = torch.as_tensor(stacked, dtype=torch.float32, device=device)
    if weights is None:
        weight_tensor = torch.ones(matrix_tensor.shape[0], device=device)
    else:
        weight_tensor = torch.as_tensor(
            list(weights), dtype=torch.float32, device=device
        )
        if weight_tensor.shape[0] != matrix_tensor.shape[0]:
            raise ValueError(
                f"weights has length {weight_tensor.shape[0]} but the panel has "
                f"{matrix_tensor.shape[0]} proteases"
            )
    return ProteasePanel(matrix_tensor, weight_tensor, codes)


class CleavagePotential(MutationPotential):
    r"""Proteolytic-susceptibility potential :math:`\Phi_{cleav}` (CLASP, Section 3).

    A protease :math:`\pi` cuts a bond when the eight-residue window around it
    (``P4 P3 P2 P1 | P1' P2' P3' P4'``) matches its specificity matrix
    :math:`M_\pi`. Given the decoder's position-factorized residue distribution
    :math:`P`, the expected per-bond score is exact. Three variants are
    provided, all aggregated over the panel with weights :math:`\rho_\pi`:

    * ``"additive"`` -- within-window expected log-score summed over bonds:
      :math:`\Phi = \sum_\pi \rho_\pi \sum_b \sum_p \sum_a P_{i(b,p)}[a] M_\pi[p,a]`.
    * ``"product"`` (default, exact rate) -- competing per-bond rates:
      :math:`R = \sum_\pi \rho_\pi \sum_b \prod_p (\sum_a P_{i(b,p)}[a] e^{M_\pi[p,a]})`,
      :math:`\Phi_{cleav} = \log R`.
    * ``"meanfield"`` -- soft maximum over bonds:
      :math:`\sum_\pi \rho_\pi\, \tau \log \sum_b e^{\mathbb{E}[s_\pi(b)]/\tau}`.

    Larger :math:`\Phi_{cleav}` means more susceptible (less stable). The
    matrices are MEROPS ``matrices_logprob.npy`` log-probabilities, so
    :math:`e^{M_\pi[p,a]}` is a residue probability.
    """

    def __init__(
        self,
        panel: ProteasePanel,
        variant: str = "product",
        alphabet: list[str] | None = None,
        background: torch.Tensor | Sequence[float] | None = None,
        temperature: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        """Initialize the cleavage potential from a protease panel.

        :param panel: Proteases with specificity matrices and weights.
        :param variant: ``"additive"``, ``"product"``, or ``"meanfield"``.
        :param alphabet: Optional index-to-token mapping for decoder residues.
        :param background: Residue distribution :math:`q_a` over the 20 amino
            acids used for subsites that fall outside the peptide. Defaults to
            uniform.
        :param temperature: Softmax temperature :math:`\\tau` for the mean-field
            variant.
        :param device: Device for computation.
        :raises ValueError: If ``variant`` is unsupported.
        """
        if variant not in {"additive", "product", "meanfield"}:
            raise ValueError(
                "variant must be 'additive', 'product', or 'meanfield'"
            )
        self.variant = variant
        self.alphabet = alphabet or DEFAULT_ALPHABET
        self.device = torch.device(device)
        self.temperature = float(temperature)
        self.codes = panel.codes

        self.matrices = panel.matrices.to(self.device)  # (N, 8, 20)
        self.exp_matrices = torch.exp(self.matrices)  # residue probabilities
        self.weights = panel.weights.to(self.device)  # (N,)

        # Decoder-vocabulary columns holding the 20 MEROPS amino acids, in
        # MEROPS order. For DEFAULT_ALPHABET this is simply [1, 2, ..., 20].
        self.merops_columns = torch.tensor(
            [self.alphabet.index(residue) for residue in MEROPS_AMINO_ACIDS],
            dtype=torch.long,
            device=self.device,
        )
        self._merops_index = {
            residue: column for column, residue in enumerate(MEROPS_AMINO_ACIDS)
        }
        self.subsite_offsets = torch.tensor(
            MEROPS_SUBSITE_OFFSETS, dtype=torch.long, device=self.device
        )
        if background is None:
            background_tensor = torch.full(
                (len(MEROPS_AMINO_ACIDS),), 1.0 / len(MEROPS_AMINO_ACIDS)
            )
        else:
            background_tensor = torch.as_tensor(background, dtype=torch.float32)
        self.background = background_tensor.to(self.device)

    def _bonds_log_potential(
        self, distributions: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        r"""Evaluate :math:`\Phi_{cleav}` for a batch of residue distributions.

        :param distributions: Per-position residue distributions over the 20
            MEROPS amino acids, shape ``(B, L, 20)``.
        :param lengths: Number of real residues in each sequence, shape ``(B,)``.
        :return: Panel-aggregated potential per sequence, shape ``(B,)``.
        """
        batch, grid, _ = distributions.shape
        n_bonds = max(grid - 1, 0)
        if n_bonds == 0:
            return torch.full((batch,), float("-inf"), device=self.device)

        bond_index = torch.arange(n_bonds, device=self.device)  # (Nb,)
        # Source position of every subsite for every bond: (Nb, 8).
        positions = bond_index.unsqueeze(1) + self.subsite_offsets.unsqueeze(0)
        clamped = positions.clamp(0, grid - 1)

        # Gather window distributions: (B, Nb, 8, 20).
        windows = distributions[:, clamped, :]
        # A subsite is real when its position lies inside the peptide.
        subsite_valid = (positions.unsqueeze(0) >= 0) & (
            positions.unsqueeze(0) < lengths.view(batch, 1, 1)
        )
        windows = torch.where(
            subsite_valid.unsqueeze(-1), windows, self.background
        )
        # A bond exists when both flanking residues are real.
        bond_valid = (bond_index.unsqueeze(0) + 1) < lengths.view(batch, 1)

        if self.variant == "product":
            # (B, Nb, 8, N) expected probability per subsite per protease.
            subsite_factor = torch.einsum(
                "bnpa,mpa->bnpm", windows, self.exp_matrices
            )
            bond_rate = subsite_factor.prod(dim=2)  # (B, Nb, N)
            bond_rate = bond_rate * bond_valid.unsqueeze(-1)
            per_protease = bond_rate.sum(dim=1)  # (B, N)
            total_rate = (per_protease * self.weights).sum(dim=1)  # (B,)
            return torch.log(total_rate.clamp_min(1e-12))

        # Additive and mean-field share the expected within-window log-score.
        subsite_term = torch.einsum("bnpa,mpa->bnpm", windows, self.matrices)
        bond_score = subsite_term.sum(dim=2)  # (B, Nb, N) = E[s_pi(b)]

        if self.variant == "additive":
            bond_score = bond_score * bond_valid.unsqueeze(-1)
            per_protease = bond_score.sum(dim=1)  # (B, N)
            return (per_protease * self.weights).sum(dim=1)

        # Mean-field: soft maximum over bonds per protease.
        masked = torch.where(
            bond_valid.unsqueeze(-1),
            bond_score,
            torch.full_like(bond_score, float("-inf")),
        )
        soft_max = self.temperature * torch.logsumexp(
            masked / self.temperature, dim=1
        )  # (B, N)
        return (soft_max * self.weights).sum(dim=1)

    def _sequences_to_distributions(
        self, sequences: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one-hot residue distributions and lengths for peptides."""
        lengths = [len(sequence) for sequence in sequences]
        grid = max(max(lengths, default=1), 1)
        distributions = np.zeros(
            (len(sequences), grid, len(MEROPS_AMINO_ACIDS)), dtype=np.float32
        )
        for row, sequence in enumerate(sequences):
            for position, residue in enumerate(sequence):
                column = self._merops_index.get(residue)
                if column is not None:
                    distributions[row, position, column] = 1.0
        return (
            torch.as_tensor(distributions, device=self.device),
            torch.tensor(lengths, dtype=torch.long, device=self.device),
        )

    def distribution_log_potential(self, decoder_output: torch.Tensor) -> torch.Tensor:
        r"""Return the differentiable :math:`\Phi_{cleav}` for decoder outputs.

        :param decoder_output: Decoder residue distribution ``Dec(z)`` with a
            trailing vocabulary axis, shape ``(..., L, V)`` where
            ``V == len(alphabet)``. A single ``(L, V)`` input is accepted.
        :return: Potential per sequence, shape ``(B,)``.
        :raises ValueError: If the vocabulary axis does not match ``alphabet``.
        """
        if decoder_output.ndim == 2:
            decoder_output = decoder_output.unsqueeze(0)
        if decoder_output.shape[-1] != len(self.alphabet):
            raise ValueError(
                f"decoder_output last axis is {decoder_output.shape[-1]}, "
                f"expected {len(self.alphabet)}"
            )
        distributions = decoder_output.index_select(-1, self.merops_columns)
        grid = distributions.shape[-2]
        # The decoder emits a full padded grid; every grid slot is a candidate
        # residue and padded slots simply carry negligible amino-acid mass.
        lengths = torch.full(
            (distributions.shape[0],), grid, dtype=torch.long, device=self.device
        )
        return self._bonds_log_potential(distributions, lengths)

    @torch.no_grad()
    def sequence_log_potential(self, sequences: Sequence[str]) -> torch.Tensor:
        r"""Return :math:`\Phi_{cleav}` for concrete peptide sequences.

        :param sequences: Peptide strings.
        :return: Potential per sequence, shape ``(len(sequences),)``.
        """
        if len(sequences) == 0:
            return torch.empty(0, device=self.device)
        distributions, lengths = self._sequences_to_distributions(sequences)
        return self._bonds_log_potential(distributions, lengths)

    @torch.no_grad()
    def compute(
        self,
        parent_peptide: str,
        mutations: dict[int, list[int]],
    ) -> dict[int, dict[int, float]]:
        r"""Return per-position :math:`\Delta_{cleav}` for each candidate residue.

        Each score is the change in susceptibility from a single substitution,
        :math:`\Phi_{cleav}(\text{mutant}) - \Phi_{cleav}(\text{parent})`, which
        is the position-separable first-order form used by MUTANG. Substitutions
        that reduce cleavage receive a negative score.

        :param parent_peptide: Parent peptide sequence.
        :param mutations: Candidate amino-acid indices grouped by position.
        :return: Nested per-position, per-residue :math:`\Delta_{cleav}`.
        """
        if not mutations:
            return {}
        padded_parent = parent_peptide.ljust(DEFAULT_MAX_LEN)
        length = len(parent_peptide)

        candidates: list[tuple[int, int]] = []
        sequences: list[str] = [parent_peptide]
        for position in sorted(mutations):
            for amino_acid in mutations[position]:
                residue = list(padded_parent)
                residue[position] = self.alphabet[amino_acid]
                candidates.append((position, amino_acid))
                sequences.append("".join(residue[:length]))

        potentials = self.sequence_log_potential(sequences).cpu().tolist()
        parent_potential = potentials[0]

        scores: dict[int, dict[int, float]] = {}
        for (position, amino_acid), potential in zip(candidates, potentials[1:]):
            scores.setdefault(position, {})[amino_acid] = (
                potential - parent_potential
            )
        return scores


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
        S. aureus, id 7).
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
