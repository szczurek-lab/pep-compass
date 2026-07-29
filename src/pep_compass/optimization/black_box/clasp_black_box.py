"""CLASP objective combining APEX activity with MEROPS cleavage stability."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation

from pep_compass.local_enumeration.mutation.mutation_potentials import (
    DEFAULT_APEX_INDEX, DEFAULT_MEROPS_DATASETS, build_clasp_potential)
from pep_compass.models.apex.APEX_predictor import PredictorAPEX


class ClaspBlackBox(AbstractBlackBox):
    r"""Minimize :math:`\log_2 \mathrm{MIC} + \lambda \Phi_{cleav}`.

    The objective trades predicted activity against proteolytic stability for a
    single pathogen. Both terms are "lower is better": ``log2 MIC`` is the APEX
    prediction for :attr:`apex_index`, and :math:`\Phi_{cleav}` is the MEROPS
    cleavage potential, so the sum is minimized.

    Note that :class:`ClaspPotential` negates this sum, because the MUTANG
    enumeration framework ranks candidates by descending score. Here the
    un-negated sum is returned together with ``maximize = False`` to match the
    convention of the other black boxes in this package.

    The individual terms are cached per sequence and exposed through
    :meth:`score_components_for`, which lets LE-BO tracking record them
    alongside the combined objective without evaluating APEX twice.
    """

    #: Per-candidate terms this objective can report to LE-BO tracking.
    component_fields = ("log2_mic", "cleavage", "cleavage_term")

    def __init__(
        self,
        *,
        variant: str = "product",
        lam: float = 0.1,
        merops_datasets: int | Iterable[int] = DEFAULT_MEROPS_DATASETS,
        merops_root: str | None = None,
        apex_index: int = DEFAULT_APEX_INDEX,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
        device: str = "cpu",
    ):
        """Initialize the CLASP objective.

        :param variant: Cleavage aggregation, one of ``"additive"``,
            ``"product"``, or ``"meanfield"``.
        :param lam: Activity/stability trade-off weight :math:`\\lambda`.
        :param merops_datasets: MEROPS dataset ids providing the protease
            panel. Ids 0-33 are APEX pathogen columns and 34 is Homo sapiens.
        :param merops_root: Directory holding the MEROPS datasets. Defaults to
            the repository ``data/merops``.
        :param apex_index: APEX pathogen column used for ``log2 MIC``.
        :param device: Torch device for APEX and the cleavage matrices.
        """
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        self.potential = build_clasp_potential(
            apex_predictor=PredictorAPEX(device=device, path="all"),
            dataset_ids=merops_datasets,
            merops_root=merops_root,
            variant=variant,
            lam=lam,
            apex_index=apex_index,
            device=device,
        )
        self._component_cache: dict[str, dict[str, float]] = {}
        self.maximize = False

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="CLASP",
            max_sequence_length=25,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=list("ACDEFGHIKLMNPQRSTVWY"),
            log_transform_recommended=False,
            discrete=True,
            padding_token=" ",
        )

    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        sequences = ["".join(sequence) for sequence in x]
        components = self.potential.score_components(sequences)
        self._cache_components(sequences, components)
        objective = components["log2_mic"] + components["cleavage_term"]

        return np.asarray(objective, dtype=float).reshape(-1, 1)

    def _cache_components(
        self, sequences: Sequence[str], components: dict[str, np.ndarray]
    ) -> None:
        for index, sequence in enumerate(sequences):
            self._component_cache[sequence] = {
                field: float(components[field][index])
                for field in self.component_fields
            }

    def score_components_for(
        self, sequences: Sequence[str]
    ) -> dict[str, list[float]]:
        """Return the separate CLASP terms for the given peptides.

        :param sequences: Peptide strings, usually already evaluated.
        :return: Each name in :attr:`component_fields` mapped to one value per
            input sequence, in input order.
        """
        sequences = list(sequences)
        missing = sorted({
            sequence
            for sequence in sequences
            if sequence not in self._component_cache
        })
        if missing:
            self._cache_components(
                missing, self.potential.score_components(missing)
            )

        return {
            field: [self._component_cache[sequence][field] for sequence in sequences]
            for field in self.component_fields
        }
