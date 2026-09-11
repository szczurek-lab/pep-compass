"""Tests for static whole-pipeline stability estimation."""

from pep_compass.core.estimation import estimate_pipeline_stability
from pep_compass.core.specification import (
    ComponentSpecification,
    FlowSpecification,
    PipelineSpecification,
)


def test_estimator_propagates_generator_growth_and_subset_cap() -> None:
    """Known component bounds must produce a finite peak estimate."""
    specification = PipelineSpecification(
        FlowSpecification(
            (
                ComponentSpecification(
                    "mutation_generator",
                    "mutang",
                    {"maximum_candidates": 8},
                ),
                ComponentSpecification(
                    "filter",
                    "candidate_subset",
                    {"count": 3},
                ),
            )
        )
    )

    estimate = estimate_pipeline_stability(
        specification,
        input_candidates=2,
        latent_dimension=4,
    )

    assert estimate.output_candidates_upper == 3
    assert estimate.peak_candidates_upper == 16
    assert estimate.latent_bytes_upper == 16 * 4 * 4
    assert estimate.warnings == ()
    generator = next(
        node for node in estimate.nodes if node.operation == "mutation_generator:mutang"
    )
    assert generator.input_candidates == 2
    assert generator.output_candidates == 16


def test_estimator_reports_unbounded_generator() -> None:
    """Unknown generator growth must remain explicit rather than guessed."""
    specification = PipelineSpecification(
        FlowSpecification(
            (ComponentSpecification("mutation_generator", "mutang", {}),)
        )
    )

    estimate = estimate_pipeline_stability(
        specification,
        input_candidates=1,
        latent_dimension=64,
    )

    assert estimate.output_candidates_upper is None
    assert estimate.peak_candidates_upper is None
    assert estimate.latent_bytes_upper is None
    assert estimate.warnings == ("Unbounded mutation generator: mutang",)
    assert estimate.nodes[0].uncertainty == "Unbounded mutation generator: mutang"
