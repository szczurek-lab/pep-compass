"""Unit tests for lineage-aware latent-jump analysis helpers."""

import numpy as np
import pandas as pd

from pep_compass.analysis.analysis_types.locality.latent_jump import (
    _match_sorbes_parents_by_id,
    _read_candidate_lineage,
)


def test_read_candidate_lineage_reads_aligned_mutang_flags_and_parent_ids():
    import torch

    from pep_compass.data_schemas.candidates import (
        CandidateBatch,
        ObjectField,
        OptionalField,
        TensorField,
    )

    batch = CandidateBatch(
        ["AA", "CC"],
        torch.zeros((2, 2)),
        {
            "mutation.parent_sequence": OptionalField(
                ObjectField(["AAA", None]), torch.tensor([True, False])
            ),
            "lineage.parent_candidate_id": TensorField(torch.tensor([41, -1])),
        },
    )
    is_mutang, parent_ids = _read_candidate_lineage(batch)
    assert is_mutang.tolist() == [True, False]
    assert parent_ids.tolist() == [41, -1]


def test_batch_without_mutation_metadata_is_not_a_mutant_pool():
    import torch

    from pep_compass.data_schemas.candidates import CandidateBatch

    is_mutang, parent_ids = _read_candidate_lineage(
        CandidateBatch(["AA"], torch.zeros((1, 2)))
    )
    assert is_mutang.tolist() == [False]
    assert parent_ids is None


def test_match_sorbes_parents_by_id_uses_exact_lineage_ids() -> None:
    """Lineage IDs must resolve parents without comparing floating-point latents."""
    points = pd.DataFrame(
        {
            "candidate_id": [12, 7],
            "trajectory_step": [2, 1],
            "sequence": ["AB", "AA"],
        }
    )

    parents = _match_sorbes_parents_by_id(np.asarray([7, 12, -1]), points)

    assert [
        parent["sequence"] if parent is not None else None for parent in parents
    ] == [
        "AA",
        "AB",
        None,
    ]
