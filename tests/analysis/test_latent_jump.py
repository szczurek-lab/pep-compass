"""Unit tests for lineage-aware latent-jump analysis helpers."""

import json

import numpy as np
import pandas as pd

from pep_compass.analysis.analysis_types.locality.latent_jump import (
    _match_sorbes_parents_by_id,
    _read_candidate_lineage,
)


def test_read_candidate_lineage_reads_mutang_flags_and_parent_ids(tmp_path) -> None:
    """Final field records must preserve MUTANG provenance and lineage IDs."""
    fields_path = tmp_path / "fields.jsonl"
    fields_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "candidate_index": 0,
                    "fields": {
                        "mutation.parent_sequence": {"valid": True},
                        "lineage.parent_candidate_id": 41,
                    },
                },
                {
                    "candidate_index": 1,
                    "fields": {
                        "mutation.parent_sequence": {"valid": False},
                        "lineage.parent_candidate_id": -1,
                    },
                },
            )
        ),
        encoding="utf-8",
    )

    is_mutang, parent_ids = _read_candidate_lineage(fields_path, candidate_count=2)

    assert is_mutang.tolist() == [True, False]
    assert parent_ids is not None
    assert parent_ids.tolist() == [41, -1]


def test_read_candidate_lineage_supports_legacy_fields(tmp_path) -> None:
    """Legacy results without lineage IDs must retain latent-match fallback."""
    fields_path = tmp_path / "fields.jsonl"
    fields_path.write_text(
        json.dumps(
            {
                "candidate_index": 0,
                "fields": {"mutation.parent_sequence": {"valid": True}},
            }
        ),
        encoding="utf-8",
    )

    is_mutang, parent_ids = _read_candidate_lineage(fields_path, candidate_count=1)

    assert is_mutang.tolist() == [True]
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

    assert [parent["sequence"] if parent is not None else None for parent in parents] == [
        "AA",
        "AB",
        None,
    ]
