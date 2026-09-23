r"""Proteolytic cleavage environment based on MEROPS specificity matrices.

This module is the general (exo-/endopeptidase) proteolysis *environment*:

* ``ProteasePanel`` / ``load_protease_panel`` load and concatenate MEROPS
  specificity matrices with per-protease weights.
* ``CleavagePotential`` evaluates the proteolytic-susceptibility potential
  :math:`\Phi_{cleav}` for concrete peptides or for a decoder's differentiable
  residue distribution.

It lives under ``components/helpers`` rather than under ``filters`` because the
same cleavage physics is reused by several components (the ranked CLASP scoring
objective, oracles, and analysis notebooks). It only depends on the shared
:class:`MutationPotential` base and default alphabet so that
``CleavagePotential`` can also be consumed directly as a per-position mutation
potential by ranked scorers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from pep_compass.optimization.components.filters.ranked.scoring.helpers.mutation_potentials import (  # noqa: E501
    DEFAULT_ALPHABET,
    DEFAULT_MAX_LEN,
    MutationPotential,
)

# MEROPS specificity-matrix layout (see data/merops/README.md and index.json).
# Amino-acid columns follow the APEX/MEROPS order, which is identical to
# ``DEFAULT_ALPHABET`` without the leading pad token.
MEROPS_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# Subsite (row) order of ``matrices_logprob.npy`` axis 1.
MEROPS_SUBSITES = ("P4", "P3", "P2", "P1", "P1p", "P2p", "P3p", "P4p")
# Position of each subsite relative to a bond cut between residues ``i`` and
# ``i + 1`` (the cut sits between P1 and P1'): P4 P3 P2 P1 | P1' P2' P3' P4'.
MEROPS_SUBSITE_OFFSETS = (-3, -2, -1, 0, 1, 2, 3, 4)
# Default protease panel: human serum/plasma subset (dataset id 35). Pathogen
# panels are ids 0-33 (APEX MIC columns); full Homo sapiens is id 34.
DEFAULT_MEROPS_DATASETS = (35,)


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

    The module lives at
    ``src/pep_compass/optimization/components/helpers/proteolysis`` so the
    repository root is six parents up.
    """
    return Path(__file__).resolve().parents[6] / "data" / "merops"


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
        id 34 is Homo sapiens, and id 35 is the human serum/plasma subset.
        Defaults to human serum (id 35).
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

    The ``reduction`` argument controls how the per-bond, per-protease scores
    collapse to one number (the panel weighted sum shown above is the default):

    * ``"sum"`` -- weighted sum over the whole panel (original behaviour).
    * ``"max_protease"`` -- only the single dominant protease
      :math:`\max_\pi \rho_\pi(\cdot)`, i.e. the most aggressive enzyme rather
      than the summed panel; bonds are still combined per ``variant``.
    * ``"max_cut"`` -- the single most-likely cut event,
      :math:`\max_\pi \max_b \rho_\pi R(\pi, b)`.
    """

    def __init__(
        self,
        panel: ProteasePanel,
        variant: str = "product",
        alphabet: list[str] | None = None,
        background: torch.Tensor | Sequence[float] | None = None,
        temperature: float = 1.0,
        device: str | torch.device = "cpu",
        reduction: str = "sum",
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
        :param reduction: How to aggregate the per-bond, per-protease scores
            into a single susceptibility. One of:

            * ``"sum"`` (default) -- weighted sum over the whole panel,
              :math:`\\sum_\\pi \\rho_\\pi(\\cdot)`, with bonds combined as the
              ``variant`` dictates. This is the original panel-wide behaviour.
            * ``"max_protease"`` -- keep only the single dominant protease,
              :math:`\\max_\\pi \\rho_\\pi(\\cdot)`, so susceptibility is set by
              the most aggressive enzyme; bonds are still combined per
              ``variant``.
            * ``"max_cut"`` -- the single most-likely cut event: maximum over
              both proteases and bonds, :math:`\\max_\\pi \\max_b \\rho_\\pi
              R(\\pi, b)`. This overrides the variant's over-bond aggregation.
        :raises ValueError: If ``variant`` or ``reduction`` is unsupported.
        """
        if variant not in {"additive", "product", "meanfield"}:
            raise ValueError(
                "variant must be 'additive', 'product', or 'meanfield'"
            )
        if reduction not in {"sum", "max_protease", "max_cut"}:
            raise ValueError(
                "reduction must be 'sum', 'max_protease', or 'max_cut'"
            )
        self.variant = variant
        self.reduction = reduction
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
            if self.reduction == "max_cut":
                # Single strongest cut: max over bonds (invalid bonds are 0).
                per_protease = bond_rate.max(dim=1).values  # (B, N)
            else:
                per_protease = bond_rate.sum(dim=1)  # (B, N)
            total_rate = self._reduce_over_panel(per_protease)  # (B,)
            return torch.log(total_rate.clamp_min(1e-12))

        # Additive and mean-field share the expected within-window log-score.
        subsite_term = torch.einsum("bnpa,mpa->bnpm", windows, self.matrices)
        bond_score = subsite_term.sum(dim=2)  # (B, Nb, N) = E[s_pi(b)]
        masked_bonds = torch.where(
            bond_valid.unsqueeze(-1),
            bond_score,
            torch.full_like(bond_score, float("-inf")),
        )

        if self.reduction == "max_cut":
            # Single strongest cut: hard max over bonds regardless of variant.
            per_protease = masked_bonds.max(dim=1).values  # (B, N)
        elif self.variant == "additive":
            per_protease = (bond_score * bond_valid.unsqueeze(-1)).sum(dim=1)
        else:
            # Mean-field: soft maximum over bonds per protease.
            per_protease = self.temperature * torch.logsumexp(
                masked_bonds / self.temperature, dim=1
            )  # (B, N)
        return self._reduce_over_panel(per_protease)

    def _reduce_over_panel(self, per_protease: torch.Tensor) -> torch.Tensor:
        r"""Aggregate weighted per-protease scores into one value per sequence.

        ``"sum"`` returns the weighted panel sum
        :math:`\sum_\pi \rho_\pi(\cdot)`; ``"max_protease"`` and ``"max_cut"``
        return the single dominant weighted protease
        :math:`\max_\pi \rho_\pi(\cdot)`.

        :param per_protease: Per-protease scores, shape ``(B, N)``.
        :return: Aggregated score per sequence, shape ``(B,)``.
        """
        weighted = per_protease * self.weights  # (B, N)
        if self.reduction == "sum":
            return weighted.sum(dim=1)
        return weighted.max(dim=1).values

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
    def sequence_bond_contributions(
        self, sequences: Sequence[str]
    ) -> list[np.ndarray]:
        r"""Return per-bond contributions that aggregate into :math:`\Phi_{cleav}`.

        For the ``"product"`` variant each entry is the panel-weighted bond rate
        :math:`\sum_\pi \rho_\pi R_{\pi,b}` (so :math:`\Phi = \log \sum_b`).
        For ``"additive"`` / ``"meanfield"`` each entry is the panel-weighted
        expected bond score :math:`\sum_\pi \rho_\pi \mathbb{E}[s_\pi(b)]`.

        :param sequences: Peptide strings.
        :return: One length-``(L-1,)`` array per sequence (empty if ``L < 2``).
        """
        if len(sequences) == 0:
            return []
        distributions, lengths = self._sequences_to_distributions(sequences)
        batch, grid, _ = distributions.shape
        n_bonds = max(grid - 1, 0)
        if n_bonds == 0:
            return [np.zeros(0, dtype=np.float64) for _ in sequences]

        bond_index = torch.arange(n_bonds, device=self.device)
        positions = bond_index.unsqueeze(1) + self.subsite_offsets.unsqueeze(0)
        clamped = positions.clamp(0, grid - 1)
        windows = distributions[:, clamped, :]
        subsite_valid = (positions.unsqueeze(0) >= 0) & (
            positions.unsqueeze(0) < lengths.view(batch, 1, 1)
        )
        windows = torch.where(
            subsite_valid.unsqueeze(-1), windows, self.background
        )
        bond_valid = (bond_index.unsqueeze(0) + 1) < lengths.view(batch, 1)

        if self.variant == "product":
            subsite_factor = torch.einsum(
                "bnpa,mpa->bnpm", windows, self.exp_matrices
            )
            bond_rate = subsite_factor.prod(dim=2) * bond_valid.unsqueeze(-1)
            contrib = (bond_rate * self.weights).sum(dim=-1)  # (B, Nb)
        else:
            subsite_term = torch.einsum("bnpa,mpa->bnpm", windows, self.matrices)
            bond_score = subsite_term.sum(dim=2) * bond_valid.unsqueeze(-1)
            contrib = (bond_score * self.weights).sum(dim=-1)

        contrib_np = contrib.detach().cpu().numpy()
        return [
            contrib_np[i, : max(int(lengths[i].item()) - 1, 0)].astype(np.float64)
            for i in range(batch)
        ]

    @torch.no_grad()
    def sequence_bond_protease_rates(
        self, sequences: Sequence[str]
    ) -> tuple[list[np.ndarray], list[str]]:
        r"""Return local MEROPS values for every bond and protease.

        For ``product`` each array contains unweighted ``R_(pi,b)`` values;
        other variants return their corresponding local expected scores.
        Arrays have shape ``(number_of_bonds, number_of_proteases)``.

        :param sequences: Peptide strings.
        :return: Per-sequence arrays and matching protease codes.
        """
        if not sequences:
            return [], list(self.codes)
        distributions, lengths = self._sequences_to_distributions(sequences)
        batch, grid, _ = distributions.shape
        n_bonds = max(grid - 1, 0)
        if n_bonds == 0:
            empty = [np.zeros((0, self.matrices.shape[0])) for _ in sequences]
            return empty, list(self.codes)
        bond_index = torch.arange(n_bonds, device=self.device)
        positions = bond_index.unsqueeze(1) + self.subsite_offsets.unsqueeze(0)
        clamped = positions.clamp(0, grid - 1)
        windows = distributions[:, clamped, :]
        subsite_valid = (positions.unsqueeze(0) >= 0) & (
            positions.unsqueeze(0) < lengths.view(batch, 1, 1)
        )
        windows = torch.where(
            subsite_valid.unsqueeze(-1), windows, self.background
        )
        bond_valid = (bond_index.unsqueeze(0) + 1) < lengths.view(batch, 1)
        if self.variant == "product":
            subsite_factor = torch.einsum(
                "bnpa,mpa->bnpm", windows, self.exp_matrices
            )
            values = subsite_factor.prod(dim=2)  # (B, Nb, N)
        else:
            subsite_term = torch.einsum("bnpa,mpa->bnpm", windows, self.matrices)
            values = subsite_term.sum(dim=2)  # (B, Nb, N)
        values = values * bond_valid.unsqueeze(-1)
        values_np = values.detach().cpu().numpy().astype(np.float64)
        return [values_np[i, : max(int(lengths[i]), 0) - 1] for i in range(batch)], list(
            self.codes
        )

    @torch.no_grad()
    def sequence_protease_rates(
        self, sequences: Sequence[str]
    ) -> tuple[np.ndarray, list[str]]:
        r"""Return per-protease total bond rates (or scores) for each sequence.

        For ``"product"`` the entry is :math:`R_\pi = \sum_b R_{\pi,b}` (so a
        single-protease potential is ``log R_π``). For ``"additive"`` /
        ``"meanfield"`` the entry is :math:`\sum_b \mathbb{E}[s_\pi(b)]`.
        Panel weights are **not** applied here so enzymes can be compared
        one-at-a-time.

        :param sequences: Peptide strings.
        :return: ``(rates, codes)`` with ``rates`` shape ``(len(sequences), N)``.
        """
        if len(sequences) == 0:
            return np.zeros((0, self.matrices.shape[0]), dtype=np.float64), list(
                self.codes
            )
        distributions, lengths = self._sequences_to_distributions(sequences)
        batch, grid, _ = distributions.shape
        n_bonds = max(grid - 1, 0)
        if n_bonds == 0:
            return (
                np.zeros((batch, self.matrices.shape[0]), dtype=np.float64),
                list(self.codes),
            )

        bond_index = torch.arange(n_bonds, device=self.device)
        positions = bond_index.unsqueeze(1) + self.subsite_offsets.unsqueeze(0)
        clamped = positions.clamp(0, grid - 1)
        windows = distributions[:, clamped, :]
        subsite_valid = (positions.unsqueeze(0) >= 0) & (
            positions.unsqueeze(0) < lengths.view(batch, 1, 1)
        )
        windows = torch.where(
            subsite_valid.unsqueeze(-1), windows, self.background
        )
        bond_valid = (bond_index.unsqueeze(0) + 1) < lengths.view(batch, 1)

        if self.variant == "product":
            subsite_factor = torch.einsum(
                "bnpa,mpa->bnpm", windows, self.exp_matrices
            )
            bond_rate = subsite_factor.prod(dim=2) * bond_valid.unsqueeze(-1)
            per_protease = bond_rate.sum(dim=1)  # (B, N)
        else:
            subsite_term = torch.einsum("bnpa,mpa->bnpm", windows, self.matrices)
            bond_score = subsite_term.sum(dim=2) * bond_valid.unsqueeze(-1)
            per_protease = bond_score.sum(dim=1)

        return per_protease.detach().cpu().numpy().astype(np.float64), list(
            self.codes
        )

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
