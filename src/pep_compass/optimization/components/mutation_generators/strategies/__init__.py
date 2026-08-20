"""Built-in mutation generator strategies."""

from pep_compass.optimization.components.mutation_generators.strategies.mutang import Mutang, MutangGenerator
from pep_compass.optimization.components.mutation_generators.strategies.mutang.combination.baseline import BaselineCombination
from pep_compass.optimization.components.mutation_generators.strategies.mutang.direction_selection.baseline import BaselineDirectionSelection
from pep_compass.optimization.components.mutation_generators.strategies.mutang.geometry.kappa_stable import KappaStableGeometry
from pep_compass.optimization.components.mutation_generators.strategies.mutang.geometry.shared import SharedGeometry
from pep_compass.optimization.components.mutation_generators.strategies.mutang.mutation_selection.threshold import ThresholdSelection
from pep_compass.optimization.components.mutation_generators.strategies.mutang.scoring.max_absolute_loading import MaxAbsoluteLoading
from pep_compass.optimization.components.mutation_generators.manager import MutationGeneratorManager
from pep_compass.optimization.components.mutation_generators.strategies.mutang.registry import (
    combination_registry,
    direction_selection_registry,
    geometry_registry,
    mutation_selection_registry,
    scoring_registry,
)
from pep_compass.utils.strategy_factory import parameter_contract


@geometry_registry.register("shared")
def build_shared_geometry() -> SharedGeometry:
    """Build geometry shared with the current SORBES point."""
    return SharedGeometry()


@geometry_registry.register("kappa_stable")
def build_kappa_stable_geometry(*, kappa: float = 1e-6) -> KappaStableGeometry:
    """Build independently computed stable geometry."""
    return KappaStableGeometry(kappa=kappa)


@direction_selection_registry.register("baseline")
def build_baseline_direction_selection(
    *, threshold: float = 1e-3, minimum: int = 5
) -> BaselineDirectionSelection:
    """Build baseline MUTANG singular-direction selection."""
    return BaselineDirectionSelection(threshold, minimum)


@scoring_registry.register("max_absolute_loading")
def build_max_absolute_loading(
    *, max_len: int = 25, alphabet_size: int = 21
) -> MaxAbsoluteLoading:
    """Build baseline per-token directional loading scorer."""
    return MaxAbsoluteLoading(max_len, alphabet_size)


@mutation_selection_registry.register("threshold")
def build_threshold_mutation_selection(*, threshold: float = 0.1) -> ThresholdSelection:
    """Build threshold-based MUTANG option selection."""
    return ThresholdSelection(threshold)


@combination_registry.register("baseline")
def build_baseline_combination(
    *, alphabet: list[str], maximum_candidates: int | None = None
) -> BaselineCombination:
    """Build baseline mutation combination enumeration."""
    return BaselineCombination(alphabet, maximum_candidates)


@MutationGeneratorManager.register("mutang")
@parameter_contract(
    accepted={
        "maximum_candidates",
        "strategies",
        "max_len",
        "direction_significance_threshold",
        "min_number_of_directions",
        "token_threshold",
        "alphabet",
    }
)
def build_mutang(autoencoder, maximum_candidates=None, strategies=None, max_len=25,
                 direction_significance_threshold=1e-3, min_number_of_directions=5,
                 token_threshold=0.1, alphabet=None):
    """Build baseline MUTANG with optional geometry selection."""
    alphabet = alphabet or list(" ACDEFGHIKLMNPQRSTVWY")
    configuration = strategies or {}
    geometry_config = configuration.get("geometry", {"method": "shared", "parameters": {}})
    directions_config = configuration.get("direction_selection", {
        "method": "baseline",
        "parameters": {
            "threshold": direction_significance_threshold,
            "minimum": min_number_of_directions,
        },
    })
    scoring_config = configuration.get("scoring", {
        "method": "max_absolute_loading",
        "parameters": {"max_len": max_len, "alphabet_size": len(alphabet)},
    })
    selection_config = configuration.get("mutation_selection", {
        "method": "threshold", "parameters": {"threshold": token_threshold},
    })
    combination_config = configuration.get("combination", {
        "method": "baseline",
        "parameters": {"alphabet": alphabet, "maximum_candidates": maximum_candidates},
    })
    mutang = Mutang(
        geometry_registry.build(
            geometry_config["method"], geometry_config.get("parameters", {})
        ),
        direction_selection_registry.build(
            directions_config["method"], directions_config.get("parameters", {})
        ),
        scoring_registry.build(
            scoring_config["method"], scoring_config.get("parameters", {})
        ),
        mutation_selection_registry.build(
            selection_config["method"], selection_config.get("parameters", {})
        ),
        combination_registry.build(
            combination_config["method"], combination_config.get("parameters", {})
        ),
        autoencoder,
    )
    return MutangGenerator(mutang)

__all__ = ["MutangGenerator"]
