"""Registration of built-in ranked and direct filters."""

from pep_compass.optimization.components.filters.direct.structural.deduplicate import DeduplicateFilter
from pep_compass.optimization.components.filters.direct.structural.candidate_subset import (
    CandidateSubsetSelector,
)
from pep_compass.optimization.components.filters.direct.optimization.robot import RobotSelector
from pep_compass.optimization.components.filters.direct.optimization.trust_region import (
    TrustRegionSelector,
    TrustRegionUpdater,
)
from pep_compass.optimization.components.filters.manager import FilterManager
from pep_compass.optimization.components.filters.direct.controls.random_lebo import RandomLeBoFilter
from pep_compass.optimization.components.filters.direct.constraints.levenshtein import LevenshteinConstraint
from pep_compass.optimization.components.filters.direct.constraints.sequence_length import SequenceLengthConstraint
from pep_compass.optimization.components.filters.ranked.base import RankedFilter
from pep_compass.optimization.components.filters.ranked.scoring.latent_geometry.lams import LamsScore
from pep_compass.optimization.components.filters.ranked.scoring.latent_geometry.move import MoveScore
from pep_compass.optimization.components.filters.ranked.scoring.latent_geometry.tandem import TandemScore
from pep_compass.optimization.components.filters.ranked.scoring.model_scores.decoder_likelihood import DecoderLikelihoodScore
from pep_compass.optimization.components.filters.ranked.scoring.model_scores.esm import ESM2PPLScorer
from pep_compass.optimization.components.filters.ranked.selection.nucleus import NucleusSelection
from pep_compass.optimization.components.filters.ranked.selection.threshold import ThresholdSelection
from pep_compass.optimization.components.filters.ranked.selection.top_k import TopKSelection
from pep_compass.optimization.components.filters.ranked.registry import (
    scoring_registry,
    selection_registry,
)
from pep_compass.utils.strategy_factory import parameter_contract


@scoring_registry.register("tandem", services={"autoencoder"})
def build_tandem_score(
    autoencoder, *, horizontal_threshold=0.1, maximum_candidates=30000, alphabet=None
) -> TandemScore:
    """Build TANDEM latent-geometry scoring."""
    return TandemScore(autoencoder, horizontal_threshold, maximum_candidates, alphabet)


@scoring_registry.register("lams", services={"autoencoder"})
def build_lams_score(
    autoencoder, *, horizontal_threshold=0.1, maximum_candidates=30000, alphabet=None
) -> LamsScore:
    """Build LAMS latent-geometry scoring."""
    return LamsScore(autoencoder, horizontal_threshold, maximum_candidates, alphabet)


@scoring_registry.register("move", services={"autoencoder"})
def build_move_score(
    autoencoder, *, maximum_candidates=30000, alphabet=None
) -> MoveScore:
    """Build MOVE latent-geometry scoring."""
    return MoveScore(autoencoder, maximum_candidates, alphabet)


@scoring_registry.register("decoder_likelihood", services={"autoencoder"})
def build_decoder_likelihood_score(
    autoencoder, *, maximum_candidates=30000, alphabet=None
) -> DecoderLikelihoodScore:
    """Build decoder-likelihood scoring."""
    return DecoderLikelihoodScore(autoencoder, maximum_candidates, alphabet)


@scoring_registry.register("esm")
def build_esm_score(
    *, model_name="esm2_t6_8M_UR50D", device="cpu"
) -> ESM2PPLScorer:
    """Build ESM pseudo-log-likelihood scoring."""
    return ESM2PPLScorer(model_name=model_name, device=device)


@selection_registry.register("threshold")
def build_threshold_selection(*, threshold: float) -> ThresholdSelection:
    """Build threshold score selection."""
    return ThresholdSelection(threshold)


@selection_registry.register("nucleus")
def build_nucleus_selection(*, top_p=0.9, temperature=1.0) -> NucleusSelection:
    """Build nucleus score selection."""
    return NucleusSelection(top_p, temperature)


@selection_registry.register("top_k")
def build_top_k_selection(*, k: int) -> TopKSelection:
    """Build top-k score selection."""
    return TopKSelection(k)


@FilterManager.register("ranked")
def build_ranked(autoencoder, scoring, selection):
    """Build an explicit score-and-selection filter composition."""
    scorer = scoring_registry.build(
        scoring["method"], scoring.get("parameters", {}),
        services={"autoencoder": autoencoder},
    )
    rule = selection_registry.build(
        selection["method"], selection.get("parameters", {}),
    )
    return RankedFilter(scorer, rule)


@FilterManager.register("esm_plausibility")
def build_esm_plausibility(
    *, threshold=None, top_k=None, model_name="esm2_t6_8M_UR50D", device="cpu"
):
    """Build ESM scoring with exactly one threshold or top-k rule."""
    if (threshold is None) == (top_k is None):
        raise ValueError("ESM filter requires exactly one of threshold or top_k.")
    selection = ThresholdSelection(threshold) if threshold is not None else TopKSelection(top_k)
    return RankedFilter(ESM2PPLScorer(model_name=model_name, device=device), selection)


@FilterManager.register("lpbebo")
@parameter_contract(accepted={"top_p", "temperature", "maximum_candidates", "alphabet"})
def build_lpbebo(autoencoder, **parameters):
    """Build the LPBEBO mutation-choice filter."""
    top_p = parameters.pop("top_p", 0.9)
    temperature = parameters.pop("temperature", 1.0)
    return RankedFilter(
        DecoderLikelihoodScore(autoencoder, **parameters),
        NucleusSelection(top_p, temperature),
    )


@FilterManager.register("lams")
@parameter_contract(
    accepted={"similarity_threshold", "horizontal_threshold", "maximum_candidates", "alphabet"}
)
def build_lams(autoencoder, **parameters):
    """Build the LAMS mutation-choice filter."""
    threshold = parameters.pop("similarity_threshold", 0.15)
    return RankedFilter(LamsScore(autoencoder, **parameters), ThresholdSelection(threshold))


@FilterManager.register("tandem")
def build_tandem(autoencoder, **parameters):
    """Build the TANDEM mutation-choice filter."""
    top_p = parameters.pop("top_p", 0.9)
    temperature = parameters.pop("temperature", 1.0)
    return RankedFilter(
        TandemScore(autoencoder, **parameters), NucleusSelection(top_p, temperature)
    )


@FilterManager.register("move")
def build_move(autoencoder, **parameters):
    """Build the MOVE mutation-choice filter."""
    top_p = parameters.pop("top_p", 0.6)
    temperature = parameters.pop("temperature", 1.0)
    return RankedFilter(
        MoveScore(autoencoder, **parameters), NucleusSelection(top_p, temperature)
    )


@FilterManager.register("random_walker")
def build_random_walker(**parameters):
    """Build random walker-mode mutation selection."""
    return RandomLeBoFilter(mode="walker", **parameters)


@FilterManager.register("random_mutang")
def build_random_mutang(**parameters):
    """Build random MUTANG-mode mutation selection."""
    return RandomLeBoFilter(mode="mutang_random", **parameters)

__all__ = [
    "CandidateSubsetSelector",
    "DeduplicateFilter",
    "RobotSelector",
    "TrustRegionSelector",
    "TrustRegionUpdater",
]
