"""Tests for translation from runtime mappings to core specifications."""

from pep_compass.core.specification import LocalEnumerationSpecification, LoopSpecification
from pep_compass.runtime.configuration.pipeline import parse_pipeline_specification


def test_pipeline_parser_builds_typed_nested_declarations() -> None:
    """Runtime syntax parsing must end at the neutral core specification."""
    specification = parse_pipeline_specification(
        {
            "steps": [
                {
                    "loop": {
                        "iterations": 2,
                        "steps": [],
                    }
                }
            ]
        }
    )

    assert isinstance(specification.root.steps[0], LoopSpecification)
    assert specification.root.steps[0].iterations == 2


def test_pipeline_parser_builds_local_enumeration_declaration() -> None:
    """Local enumeration must retain ordered local filters in its declaration."""
    specification = parse_pipeline_specification(
        {
            "steps": [
                {
                    "local_enumeration": {
                        "trajectories": 2,
                        "trajectory_execution": "sequential",
                        "iterations": 3,
                        "walker": {"method": "sorbes", "parameters": {}},
                        "mutation_generator": {
                            "method": "mutang",
                            "parameters": {},
                        },
                        "filters": [
                            {"method": "deduplicate", "parameters": {}}
                        ],
                    }
                }
            ]
        }
    )

    operation = specification.root.steps[0]
    assert isinstance(operation, LocalEnumerationSpecification)
    assert operation.trajectories == 2
    assert operation.trajectory_execution == "sequential"
    assert [step.method for step in operation.filters.steps] == ["deduplicate"]
