"""Factories for optimization models and algorithm components."""

from __future__ import annotations

from typing import Any


def _build_encoder(config: dict[str, Any]):
    """Construct the configured HydrAMP encoder-decoder lazily.

    :param config: Resolved task configuration.
    :return: HydrAMP model placed on ``config["device"]``.
    """
    import torch

    from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
        HydrAMPEncoderDecoder,
    )

    encoder = config["encoder"]
    return HydrAMPEncoderDecoder(
        jacobian_mode=encoder["jacobian_mode"],
        device=config["device"],
        default_condition=torch.tensor(
            encoder["default_condition"], device=config["device"]
        ),
        temp=encoder["temperature"],
        jacobian_eps=encoder["jacobian_eps"],
        field_eps=encoder["field_eps"],
    )


def build_black_box(config: dict[str, Any]):
    """Construct the selected discrete biological objective lazily.

    :param config: Resolved task configuration.
    :return: Registered POLI-compatible black box.
    """
    black_box = config["black_box"]
    name = black_box["name"]
    common = black_box.get("common", {})
    if name == "apex":
        from pep_compass.optimization.black_box.apex_black_box import APEXBlackBox

        return APEXBlackBox(
            mic_aggregate=black_box["apex"]["mic_aggregate"],
            mic_bacteria=black_box["apex"]["mic_bacteria"],
            device=config["device"],
            **common,
        )
    if name == "clasp":
        from pep_compass.optimization.black_box.clasp_black_box import ClaspBlackBox

        clasp = black_box["clasp"]
        return ClaspBlackBox(
            variant=clasp["variant"],
            lam=clasp["lam"],
            merops_datasets=clasp["merops_datasets"],
            merops_root=clasp["merops_root"],
            apex_index=clasp["apex_index"],
            device=config["device"],
            **common,
        )
    if name == "battleamp":
        from pep_compass.optimization.black_box.battleamp_black_box import (
            BattleAMPBlackBox,
        )

        return BattleAMPBlackBox(device=config["device"], **common)
    if name == "hydrophobicity":
        from pep_compass.optimization.black_box.hydrophobicity_black_box import (
            HydrophobicityBlackBox,
        )

        return HydrophobicityBlackBox(
            scale=black_box["hydrophobicity"]["scale"], **common
        )
    if name == "toxipep":
        from pep_compass.optimization.black_box.toxipep_black_box import ToxiPepBlackBox

        return ToxiPepBlackBox(device=config["device"], **common)
    raise AssertionError(f"Validated black box is not implemented: {name}")


def _build_mutation_enumerator(
    config: dict[str, Any],
):
    """Construct canonical MUTANG from the mutation configuration.

    :param config: Resolved task configuration.
    :return: Tangent-space mutation enumerator.
    """
    from pep_compass.local_enumeration.mutation_enumerator import (
        MutationEnumerationInTangentSpace,
    )

    mutation = config["mutation"]
    return MutationEnumerationInTangentSpace(
        max_len=mutation["max_len"],
        direction_significance_threshold=mutation["direction_significance_threshold"],
        min_number_of_directions=mutation["min_number_of_directions"],
        token_threshold=mutation["token_threshold"],
    )


def _build_walker(
    config: dict[str, Any], encoder_decoder
):
    """Construct canonical SORBES for the supplied encoder-decoder.

    :param config: Resolved task configuration.
    :param encoder_decoder: Model providing decoder geometry.
    :return: Configured second-order SORBES walker.
    """
    from pep_compass.local_enumeration.sampling_walker import (
        SecondOrderRiemannianBrownianEfficientSampling,
    )

    walker = config["walker"]
    return SecondOrderRiemannianBrownianEfficientSampling(
        encoder_decoder=encoder_decoder,
        horizontal_threshold=walker["horizontal_threshold"],
        time_step=walker["time_step"],
        max_horizontal_update_norm=walker["max_horizontal_update_norm"],
        vertical_movement=walker["vertical_movement"],
    )


def _build_filter(config: dict[str, Any], encoder_decoder):
    """Construct the method-specific LE-BO candidate filter.

    :param config: Resolved task configuration.
    :param encoder_decoder: Model used by probability or geometry filters.
    :return: Filter registered for the configured candidate strategy.
    :raises ValueError: If the selected method has no filter.
    """
    from pep_compass.local_enumeration.mutation.mutation_filters import (
        LamsFilter,
        LpbeboFilter,
        MoveFilter,
        RandomLeBoFilter,
        TandemFilter,
    )

    filter_config = config["filter"]
    candidate_strategy = config["optimizer"]["lebo"]["candidate_strategy"]
    common = {"maximum_candidates": filter_config["maximum_candidates"]}
    if candidate_strategy == "lpbebo":
        return LpbeboFilter(
            encoder_decoder,
            top_p=filter_config["top_p"],
            temperature=filter_config["temperature"],
            **common,
        )
    if candidate_strategy == "lams":
        return LamsFilter(
            encoder_decoder,
            horizontal_threshold=config["walker"]["horizontal_threshold"],
            similarity_threshold=filter_config["similarity_threshold"],
            **common,
        )
    if candidate_strategy == "tandem":
        return TandemFilter(
            encoder_decoder,
            horizontal_threshold=config["walker"]["horizontal_threshold"],
            top_p=filter_config["top_p"],
            temperature=filter_config["temperature"],
            **common,
        )
    if candidate_strategy == "move":
        return MoveFilter(
            encoder_decoder,
            top_p=filter_config["top_p"],
            temperature=filter_config["temperature"],
            **common,
        )
    if candidate_strategy in {"random_walker", "random_mutang"}:
        return RandomLeBoFilter(
            mode=(
                "walker"
                if candidate_strategy == "random_walker"
                else "mutang_random"
            ),
            selection_fraction=filter_config["selection_fraction"],
            maximum_positions=filter_config["maximum_positions"],
            residues_per_position=filter_config["residues_per_position"],
            **common,
        )
    raise ValueError(
        f"LE-BO candidate strategy {candidate_strategy!r} does not define a filter"
    )


def _build_local_enumerator(
    config: dict[str, Any], encoder_decoder
):
    """Compose SORBES, MUTANG, and the optional method filter.

    :param config: Resolved LE-BO task configuration.
    :param encoder_decoder: HydrAMP model shared by local components.
    :return: Local enumerator matching the selected method.
    """
    from pep_compass.local_enumeration.local_enumerator import (
        FilteredMutationLocalEnumerator,
        SamplingFilteredMutationLocalEnumerator,
        SamplingMutationLocalEnumerator,
    )

    mutation_enumerator = _build_mutation_enumerator(config)
    local = config["local_enumeration"]
    candidate_strategy = config["optimizer"]["lebo"]["candidate_strategy"]
    if candidate_strategy == "lpbebo":
        return FilteredMutationLocalEnumerator(
            encoder_decoder=encoder_decoder,
            mutation_generator=mutation_enumerator,
            candidate_filter=_build_filter(config, encoder_decoder),
            max_neighbour_levenstein=local["max_neighbour_levenshtein"],
            device=config["device"],
            tracking_level=config["tracking"]["level"],
        )
    common = {
        "encoder_decoder": encoder_decoder,
        "sampling_walker": _build_walker(config, encoder_decoder),
        "mutation_enumerator": mutation_enumerator,
        "walker_trajectories_number": local["walker_trajectories"],
        "time_walk_budget": local["walk_time_budget"],
        "max_neighbour_levenstein": local["max_neighbour_levenshtein"],
        "device": config["device"],
        "tracking_level": config["tracking"]["level"],
    }
    if candidate_strategy == "lebo":
        return SamplingMutationLocalEnumerator(**common)
    return SamplingFilteredMutationLocalEnumerator(
        candidate_filter=_build_filter(config, encoder_decoder), **common
    )


def _build_latent_black_box(config: dict[str, Any], discrete_black_box):
    """Wrap a discrete objective with HydrAMP latent decoding.

    :param config: Resolved task configuration.
    :param discrete_black_box: Objective evaluated after decoding.
    :return: Latent POLI black box.
    """
    from pep_compass.optimization.black_box.hydramp_black_box_wrapper import (
        HydrAMPBlackBoxWrapper,
    )

    encoder = config["encoder"]
    return HydrAMPBlackBoxWrapper(
        black_box=discrete_black_box,
        device=config["device"],
        jacobian_eps=encoder["jacobian_eps"],
        field_eps=encoder["field_eps"],
    )


def build_optimizer(config: dict[str, Any], discrete_black_box):
    """Construct the optimizer and its observed objective.

    :param config: Resolved task configuration.
    :param discrete_black_box: Selected peptide objective.
    :return: Tuple of optimizer, observed black box, and optional
        encoder-decoder used to decode observer inputs.
    :raises RuntimeError: If LaMBO2 is selected without its optional extras.
    """
    optimizer_config = config["optimizer"]
    name = optimizer_config["name"]
    if name == "lebo":
        from pep_compass.optimization.lebo.local_enumeration_bayesian_optimizer import (
            LocalEnumerationBayesianOptimizer,
        )

        encoder_decoder = _build_encoder(config)
        lebo = optimizer_config["lebo"]
        optimizer = LocalEnumerationBayesianOptimizer(
            black_box=discrete_black_box,
            local_enumerator=_build_local_enumerator(config, encoder_decoder),
            device=config["device"],
            evaluations_per_iteration=lebo["evaluations_per_iteration"],
            levenstain_diversity_threshold=lebo[
                "levenshtein_diversity_threshold"
            ],
            initial_peptides_number=lebo["initial_peptides"],
            turbo_success_tolerance=lebo["turbo_success_tolerance"],
            turbo_failure_tolerance=lebo["turbo_failure_tolerance"],
            turbo_length_init=lebo["turbo_length_init"],
            turbo_length_min=lebo["turbo_length_min"],
            turbo_length_max=lebo["turbo_length_max"],
            acquisition_batch_size=lebo["acquisition_batch_size"],
            standardize=lebo["standardize"],
            best_as_center=lebo["best_as_center"],
            blosum_diversity_matrix=lebo["blosum_diversity_matrix"],
            blosum_diversity_max_score=lebo["blosum_diversity_max_score"],
        )
        return optimizer, discrete_black_box, encoder_decoder
    if name == "random_mutation":
        from pep_compass.optimization.baselines.random_mutation import (
            RandomMutationOptimizer,
        )

        random_config = optimizer_config["random_mutation"]
        optimizer = RandomMutationOptimizer(
            black_box=discrete_black_box,
            esm_model_name=random_config["esm_model_name"],
            esm_ppl_threshold=random_config["esm_ppl_threshold"],
            esm_device=random_config["esm_device"],
            esm_max_resampling_attempts=random_config[
                "esm_max_resampling_attempts"
            ],
        )
        return optimizer, discrete_black_box, None
    if name in {"cmaes", "saasbo"}:
        import torch

        latent_black_box = _build_latent_black_box(config, discrete_black_box)
        if name == "cmaes":
            from pep_compass.optimization.baselines.latent_cmaes import (
                LatentCMAESOptimizer,
            )

            optimizer = LatentCMAESOptimizer(
                black_box=latent_black_box, device=config["device"]
            )
            cmaes = optimizer_config["cmaes"]
            optimizer.population_size = cmaes["population_size"]
            optimizer.initial_sigma = cmaes["initial_sigma"]
            optimizer.constraint_penalty = cmaes["constraint_penalty"]
        else:
            from pep_compass.optimization.baselines.saasbo import SaasboOptimizer

            saasbo = optimizer_config["saasbo"]
            optimizer = SaasboOptimizer(
                black_box=latent_black_box,
                device=torch.device(config["device"]),
                batch_size=saasbo["batch_size"],
                warmup_steps=saasbo["warmup_steps"],
                num_samples=saasbo["num_samples"],
                thinning=saasbo["thinning"],
                dim=saasbo["dimension"],
            )
        return optimizer, latent_black_box, latent_black_box.encoder_decoder
    if name == "lambo2":
        try:
            from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2
        except ImportError as error:
            raise RuntimeError(
                "LaMBO2 requires the optional poli-baselines[lambo2] dependencies"
            ) from error
        return LaMBO2, discrete_black_box, None
    raise AssertionError(f"Validated optimizer is not implemented: {name}")

