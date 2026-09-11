"""Tests for non-fatal computation-graph diagnostics."""

from pep_compass.core.specification import (
    ComponentSpecification,
    FlowSpecification,
    LoopSpecification,
    LocalEnumerationSpecification,
    PipelineSpecification,
)
from pep_compass.core.validation import diagnose_pipeline_specification


def test_diagnostics_reject_mutation_feedback_into_walker_loop() -> None:
    """A generic walker-generator loop must expose trajectory branching."""
    specification = PipelineSpecification(
        FlowSpecification(
            (
                LoopSpecification(
                    2,
                    FlowSpecification(
                        (
                            ComponentSpecification("walker", "sorbes"),
                            ComponentSpecification("mutation_generator", "mutang"),
                        )
                    ),
                ),
            )
        )
    )

    diagnostics = diagnose_pipeline_specification(specification)

    assert {diagnostic.code for diagnostic in diagnostics} == {
        "TRAJECTORY_FEEDBACK",
        "UNBOUNDED_EXPANSION",
    }


def test_local_enumeration_is_not_reported_as_generic_loop_feedback() -> None:
    """An outer loop must respect LocalEnumeration's private walker state."""
    local = LocalEnumerationSpecification(
        trajectories=1,
        iterations=1,
        walk_time=None,
        walker=ComponentSpecification("walker", "sorbes"),
        generator=ComponentSpecification(
            "mutation_generator", "mutang", {"maximum_candidates": 10}
        ),
        filters=FlowSpecification(()),
        include_walk_points=False,
    )
    specification = PipelineSpecification(
        FlowSpecification((LoopSpecification(2, FlowSpecification((local,))),))
    )

    diagnostics = diagnose_pipeline_specification(specification)

    assert "TRAJECTORY_FEEDBACK" not in {
        diagnostic.code for diagnostic in diagnostics
    }
