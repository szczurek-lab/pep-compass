"""Tests for bounded model-backed test-run transformations."""

from pep_compass.runtime.test_run import (
    TestRunPolicy as RunPolicy,
    constrain_pipeline_for_test_run,
)


def test_test_run_constrains_nested_expansion_without_mutating_source() -> None:
    """Test-run limits must be explicit and leave the loaded config unchanged."""
    source = {
        "limits": {"generated_candidates": 1000},
        "steps": [
            {
                "local_enumeration": {
                    "trajectories": 4,
                    "walk_time": 2.0,
                    "walker": {"method": "sorbes", "parameters": {}},
                    "mutation_generator": {
                        "method": "mutang",
                        "parameters": {},
                    },
                    "filters": [],
                }
            }
        ],
    }

    constrained, overrides = constrain_pipeline_for_test_run(
        source,
        RunPolicy(iterations=1, candidates=2),
    )

    settings = constrained["steps"][0]["local_enumeration"]
    assert "walk_time" not in settings
    assert settings["iterations"] == 1
    assert settings["trajectories"] == 1
    assert settings["mutation_generator"]["parameters"]["maximum_candidates"] == 2
    assert source["steps"][0]["local_enumeration"]["walk_time"] == 2.0
    assert overrides
