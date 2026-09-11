"""Latent-space displacement relative to sequence edit distance.

Measures whether MUTANG candidates that are only a few substitutions away
from a reference sequence (small Levenshtein distance) can nonetheless land
far away in latent space (large Euclidean distance) -- i.e. whether the
mutation "jumps" rather than stays local.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from pep_compass.analysis.analysis_types._encoding import encode_sequences
from pep_compass.analysis.analysis_types.locality._streaming import NestedProgress
from pep_compass.analysis.reader.selection import ExperimentSelection
from pep_compass.analysis.result import AnalysisResult
from pep_compass.autoencoder.strategies.hydramp.adapter import HydrampAutoencoder

DISTANCE_KINDS = (
    "candidate_to_origin",
    "candidate_to_sorbes_parent",
    "sorbes_to_origin",
)


def _levenshtein_distance(left: str, right: str) -> int:
    """Return the exact edit distance using one dynamic-programming row.

    Duplicated rather than imported from the runtime's ``LevenshteinConstraint``
    filter: ``pep_compass.analysis`` never depends on the optimization engine.
    """
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _resolve_autoencoder_parameters(config: dict[str, Any]) -> dict[str, Any]:
    """Extract standalone ``HydrampAutoencoder`` constructor arguments."""
    autoencoder_config = config["autoencoder"]
    if autoencoder_config["method"] != "hydramp":
        raise ValueError(
            "latent_jump only supports the hydramp autoencoder, got "
            f"{autoencoder_config['method']!r}."
        )
    parameters = autoencoder_config["parameters"]
    return {
        "jacobian_mode": parameters["jacobian_mode"],
        "jacobian_eps": parameters["jacobian_eps"],
        "field_eps": parameters["field_eps"],
        "model_name": autoencoder_config["model"],
    }


def _match_sorbes_parents(
    copied_latents: np.ndarray,
    trajectory_points: pd.DataFrame,
    trajectory_point_latents: np.ndarray,
    decimals: int,
) -> list[pd.Series | None]:
    """Identify each candidate's generating SORBES point by exact latent match.

    MUTANG never re-encodes generated sequences: ``CandidateBatch.with_sequences``
    replaces sequences "without changing their latent origins or fields", so
    every MUTANG candidate's stored latent origin is a bit-identical copy of
    the walk point that produced it. This fallback is only used for older run
    outputs without ``lineage.parent_candidate_id``; current outputs resolve
    the parent directly by ID in :func:`_match_sorbes_parents_by_id`.
    """
    lookup: dict[tuple[float, ...], pd.Series] = {}
    for position, vector in enumerate(trajectory_point_latents):
        key = tuple(np.round(vector, decimals))
        lookup.setdefault(key, trajectory_points.iloc[position])
    return [
        lookup.get(tuple(np.round(vector, decimals))) for vector in copied_latents
    ]


def _read_candidate_lineage(
    fields_path: Any,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read MUTANG provenance and optional parent IDs from final candidate fields.

    Only MUTANG output has ``mutation.parent_sequence`` attached, so its
    presence/validity flag distinguishes candidate provenance in the final
    result pool. Current outputs additionally store the exact lineage parent.
    """
    mask = np.zeros(candidate_count, dtype=bool)
    parent_ids = np.full(candidate_count, -1, dtype=np.int64)
    has_parent_ids = False
    with open(fields_path, encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            candidate_index = int(payload["candidate_index"])
            if not 0 <= candidate_index < candidate_count:
                raise ValueError(
                    "Candidate field index is outside the final candidate table: "
                    f"{candidate_index}."
                )
            marker = payload["fields"].get("mutation.parent_sequence")
            mask[candidate_index] = bool(
                isinstance(marker, dict) and marker.get("valid")
            )
            parent_id = payload["fields"].get("lineage.parent_candidate_id")
            if isinstance(parent_id, int) and not isinstance(parent_id, bool):
                parent_ids[candidate_index] = parent_id
                has_parent_ids = True
    return mask, parent_ids if has_parent_ids else None


def _match_sorbes_parents_by_id(
    parent_ids: np.ndarray,
    trajectory_points: pd.DataFrame,
) -> list[pd.Series | None]:
    """Resolve MUTANG parents against tracked SORBES candidates by lineage ID."""
    if "candidate_id" not in trajectory_points:
        raise ValueError("Trajectory points do not contain candidate lineage IDs.")
    points_by_id: dict[int, pd.Series] = {}
    for _, point in trajectory_points.iterrows():
        candidate_id = point["candidate_id"]
        if pd.isna(candidate_id):
            continue
        candidate_id = int(candidate_id)
        if candidate_id in points_by_id:
            raise ValueError(
                "Trajectory candidate IDs must be unique within a run; "
                f"found {candidate_id} more than once."
            )
        points_by_id[candidate_id] = point
    return [points_by_id.get(int(parent_id)) for parent_id in parent_ids]


def compute_outliers(
    data: pd.DataFrame,
    *,
    group_columns: list[str] = ("distance_kind", "grid_value", "levenshtein_distance"),
    outlier_iqr_multiplier: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag Tukey far outliers within each group, e.g. by edit distance or iteration.

    Split out from :func:`latent_jump` so it can be recomputed cheaply
    (plain pandas groupby, no autoencoder) directly from an already-computed
    ``result.data`` -- including a cached result, whose ``diagnostics`` are
    not persisted by ``MetricsStore`` and so are otherwise empty.

    :param data: Long-format observations with ``distance_kind``,
        ``grid_value``, ``euclidean_distance``, and every column named in
        ``group_columns``, e.g. ``latent_jump``'s ``data``.
    :param group_columns: Columns defining "the same group" an observation is
        compared against. Default groups by edit distance (answers "unusually
        far for this Levenshtein distance?"); pass
        ``("distance_kind", "grid_value", "trajectory_step")`` to instead ask
        "unusually far for this iteration?" -- e.g. to find which iterations
        produce the most extreme jumps.
    :param outlier_iqr_multiplier: An observation is an outlier when its
        ``euclidean_distance`` exceeds ``Q3 + outlier_iqr_multiplier * IQR``
        within its own group -- i.e. unusually far for that specific group,
        not just far in absolute terms.
    :return: ``(outliers, frequency)``: the flagged rows sorted by distance,
        and per-group ``candidate_count``/``outlier_count``/``outlier_fraction``.
    """
    group_columns = list(group_columns)
    thresholds = (
        data.groupby(group_columns)["euclidean_distance"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "q25", 0.75: "q75"})
    )
    thresholds["outlier_threshold"] = thresholds["q75"] + outlier_iqr_multiplier * (
        thresholds["q75"] - thresholds["q25"]
    )
    annotated = data.join(thresholds["outlier_threshold"], on=group_columns)
    is_outlier = annotated["euclidean_distance"] > annotated["outlier_threshold"]

    outliers = (
        annotated[is_outlier]
        .drop(columns=["outlier_threshold"])
        .sort_values("euclidean_distance", ascending=False)
        .reset_index(drop=True)
    )
    frequency = (
        annotated.assign(is_outlier=is_outlier)
        .groupby(group_columns)["is_outlier"]
        .agg(candidate_count="size", outlier_count="sum")
        .reset_index()
    )
    frequency["outlier_fraction"] = (
        frequency["outlier_count"] / frequency["candidate_count"]
    )
    return outliers, frequency


def rank_by_jump_ratio(
    data: pd.DataFrame,
    *,
    distance_kind: str = "candidate_to_sorbes_parent",
    minimum_levenshtein: int = 1,
) -> pd.DataFrame:
    """Rank observations by latent displacement per edit, few-mutations-first.

    :func:`compute_outliers` flags a distance as unusual only relative to
    its peers *at the same Levenshtein bin* -- a large jump at a
    high-Levenshtein bin, where everyone jumps far, won't stand out. This
    instead directly ranks "few mutations, large latent move" by computing
    ``euclidean_distance / levenshtein_distance`` per row: the higher this
    ratio, the more a single edit displaced the sequence in latent space.

    :param data: Long-format observations, e.g. ``latent_jump``'s ``data``.
    :param distance_kind: Which distance kind to rank. Default is
        ``candidate_to_sorbes_parent`` -- the marginal jump caused by this
        step's mutation alone, as opposed to ``candidate_to_origin`` (which
        also bakes in however far the walker itself already drifted).
    :param minimum_levenshtein: Exclude rows below this edit distance
        (``levenshtein_distance=0`` makes the ratio undefined).
    :return: Rows for ``distance_kind`` with a ``jump_ratio`` column, sorted
        descending -- the most disproportionate jumps first.
    """
    subset = data[
        (data["distance_kind"] == distance_kind)
        & (data["levenshtein_distance"] >= minimum_levenshtein)
    ].copy()
    subset["jump_ratio"] = (
        subset["euclidean_distance"] / subset["levenshtein_distance"]
    )
    return subset.sort_values("jump_ratio", ascending=False).reset_index(drop=True)


def _grid_value(config: dict[str, Any]) -> Any:
    """Read the swept ``direction_significance_threshold`` from a run's config."""
    return config["pipeline"]["steps"][0]["local_enumeration"]["mutation_generator"][
        "parameters"
    ]["direction_significance_threshold"]


def sorbes_trajectory_profile(selection: ExperimentSelection) -> AnalysisResult:
    """Return each SORBES walk point's latent drift from its trajectory origin.

    One row per ``(run, trajectory, iteration)``: no re-encoding, autoencoder,
    or MUTANG output involved -- purely genuine walker-produced latents from
    ``tracking/trajectory_points.csv`` / ``trajectory_latents.pt``. Intended
    to answer "which seed sequences does SORBES actually move for" before
    interpreting any MUTANG-candidate distance analysis.

    :param selection: Runs to analyze.
    :return: One row per walk point, with ``euclidean_distance`` from the
        trajectory's original seed and its own ``trajectory_step`` (iteration
        index) for the x-axis.
    """
    rows: list[dict[str, Any]] = []
    for run in selection.runs:
        replay = selection.reader.replay(run.run_id)
        local_enumerations = replay.local_enumerations
        if local_enumerations.empty:
            continue
        entry_execution_id = int(local_enumerations.iloc[0]["execution_id"])
        origin_sequences, origin_latents = replay.local_enumeration_input(
            entry_execution_id
        )
        origin_sequence = origin_sequences[0]
        origin_latent = origin_latents[0].numpy()

        points = replay.trajectory_points
        point_latents_full = replay.trajectory_latents.numpy()
        point_latents = point_latents_full[points["latent_index"].to_numpy()]
        moved = points["sequence"].nunique() > 1
        for (_, point), latent in zip(points.iterrows(), point_latents):
            rows.append(
                {
                    "run_id": run.run_id,
                    "grid_value": _grid_value(run.config),
                    "seed_sequence": origin_sequence,
                    "sorbes_moved": moved,
                    "trajectory_id": point["trajectory_id"],
                    "trajectory_index": int(point["trajectory_index"]),
                    "trajectory_step": int(point["trajectory_step"]),
                    "euclidean_distance": float(
                        np.linalg.norm(latent - origin_latent)
                    ),
                    "sequence": point["sequence"],
                }
            )
    data = pd.DataFrame(rows)
    return AnalysisResult(
        data,
        {
            "analysis": "sorbes_trajectory_profile",
            "run_count": len(selection.runs),
            "walk_point_count": len(data),
            "seeds_moved": sorted(
                data.loc[data["sorbes_moved"], "seed_sequence"].unique().tolist()
            )
            if not data.empty
            else [],
            "seeds_stationary": sorted(
                data.loc[~data["sorbes_moved"], "seed_sequence"].unique().tolist()
            )
            if not data.empty
            else [],
        },
    )


def latent_jump(
    selection: ExperimentSelection,
    *,
    levenshtein_bins: tuple[int, ...] = tuple(range(0, 9)),
    outlier_iqr_multiplier: float = 3.0,
    candidate_sample_size: int | None = 20_000,
    encode_batch_size: int = 1024,
    match_decimals: int = 5,
    random_seed: int = 0,
    device: str = "cpu",
    progress: bool = True,
) -> AnalysisResult:
    """Relate sequence edit distance to genuine latent-space displacement.

    For every selected run's final ``local_enumeration`` output, computes
    three Euclidean latent distances against Levenshtein edit distance:

    - ``candidate_to_origin``: a MUTANG candidate to its trajectory's
      original seed, both freshly re-encoded.
    - ``candidate_to_sorbes_parent``: a MUTANG candidate to the SORBES walk
      point it was generated from (identified by lineage ID, with an exact
      latent-match fallback for legacy outputs). MUTANG candidates are the *only* thing
      subject to the pipeline's ``levenshtein`` local filter (``maximum_distance``
      applied against ``local_enumeration.center_sequence``); raw SORBES walk
      points bypass it entirely (they are appended to the trajectory's output
      before ``self.filters`` runs), so this distance kind is truncated at
      that filter's radius while ``sorbes_to_origin`` is not.
    - ``sorbes_to_origin``: a SORBES walk point's drift from the trajectory's
      original seed (both are genuine walker-produced latents; no
      re-encoding needed).

    Each row also carries ``sorbes_moved``: whether that row's trajectory's
    SORBES walker ever decoded to more than one distinct sequence across all
    its walk points (see :func:`sorbes_trajectory_profile`). Some seeds barely
    move at all under a given walker configuration; pooling them with seeds
    that do changes what the aggregate distribution means.

    :param selection: Runs to analyze; must share one autoencoder configuration.
    :param levenshtein_bins: Edit-distance values retained in the result.
    :param outlier_iqr_multiplier: Tukey far-outlier multiplier applied per
        ``(distance_kind, grid_value, levenshtein_distance)`` group to flag
        candidates that jump unusually far in latent space for their bin.
    :param candidate_sample_size: Maximum MUTANG candidates re-encoded per
        run; ``None`` re-encodes every final MUTANG candidate.
    :param encode_batch_size: Sequences encoded per autoencoder forward pass.
    :param match_decimals: Rounding precision used to match a candidate's
        inherited latent to its SORBES-parent walk point for legacy runs that
        do not record lineage identifiers.
    :param random_seed: Seed for candidate subsampling.
    :param device: Torch device used only for this analysis's re-encoding.
    :param progress: Print an overall bar plus one bar per run being encoded.
    :return: Long-format per-observation distances (one row per candidate or
        walk point per distance kind) in ``data``, each carrying
        ``trajectory_step`` (the generating SORBES iteration -- for candidate
        rows, their matched parent's iteration; ``None`` if unmatched) so the
        same table supports both the by-Levenshtein-bin and by-iteration
        views, plus an outlier table and per-bin outlier frequency in
        ``diagnostics``.
    :raises ValueError: If selected runs use different autoencoder
        configurations or a non-hydramp autoencoder.
    """
    empty_metadata = {"analysis": "latent_jump", "run_count": 0}
    if not selection.runs:
        return AnalysisResult(pd.DataFrame(), empty_metadata)

    autoencoder_parameters = _resolve_autoencoder_parameters(selection.runs[0].config)
    for run in selection.runs[1:]:
        if _resolve_autoencoder_parameters(run.config) != autoencoder_parameters:
            raise ValueError(
                "latent_jump requires every selected run to share one "
                "autoencoder configuration."
            )
    autoencoder = HydrampAutoencoder(device=device, **autoencoder_parameters)
    autoencoder = autoencoder.to(device).eval()

    levenshtein_bins = tuple(sorted(set(levenshtein_bins)))
    rng = np.random.default_rng(random_seed)

    # Pre-pass: how many MUTANG candidates will actually be encoded, so the
    # overall progress bar has a real total before any encoding starts.
    planned: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for run in selection.runs:
        replay = selection.reader.replay(run.run_id)
        final_candidates = replay.final_candidates
        is_mutang, parent_ids = _read_candidate_lineage(
            replay.root / "fields.jsonl", len(final_candidates)
        )
        mutang_indices = np.flatnonzero(is_mutang)
        if (
            candidate_sample_size is not None
            and len(mutang_indices) > candidate_sample_size
        ):
            mutang_indices = rng.choice(
                mutang_indices, size=candidate_sample_size, replace=False
            )
        planned[run.run_id] = (mutang_indices, parent_ids)
    overall_total = sum(len(indices) for indices, _ in planned.values())
    nested_progress = NestedProgress("latent_jump:encode", overall_total, enabled=progress)

    rows: list[dict[str, Any]] = []
    total_walk_points = 0
    total_candidates_analyzed = 0
    unmatched_parents = 0
    lineage_parent_matches = 0
    latent_parent_matches = 0

    for run in selection.runs:
        replay = selection.reader.replay(run.run_id)
        grid_value = _grid_value(run.config)

        local_enumerations = replay.local_enumerations
        if local_enumerations.empty:
            continue
        entry_execution_id = int(local_enumerations.iloc[0]["execution_id"])
        origin_sequences, origin_latents = replay.local_enumeration_input(
            entry_execution_id
        )
        origin_sequence = origin_sequences[0]
        origin_latent = origin_latents[0].numpy()

        points = replay.trajectory_points
        point_latents_full = replay.trajectory_latents.numpy()
        point_latents = point_latents_full[points["latent_index"].to_numpy()]
        sorbes_moved = points["sequence"].nunique() > 1
        for sequence, latent, trajectory_step in zip(
            points["sequence"], point_latents, points["trajectory_step"]
        ):
            rows.append(
                {
                    "run_id": run.run_id,
                    "grid_value": grid_value,
                    "sorbes_moved": sorbes_moved,
                    "distance_kind": "sorbes_to_origin",
                    "trajectory_step": int(trajectory_step),
                    "levenshtein_distance": _levenshtein_distance(
                        sequence, origin_sequence
                    ),
                    "euclidean_distance": float(
                        np.linalg.norm(latent - origin_latent)
                    ),
                    "sequence": sequence,
                }
            )
        total_walk_points += len(points)

        mutang_indices, parent_ids = planned[run.run_id]
        if len(mutang_indices) == 0:
            continue
        total_candidates_analyzed += len(mutang_indices)

        final_candidates = replay.final_candidates
        final_latents = replay.final_latents.numpy()
        sequences = final_candidates["sequence"].to_numpy()[mutang_indices]
        copied_latents = final_latents[mutang_indices]
        if parent_ids is not None and "candidate_id" in points:
            matched_parents = _match_sorbes_parents_by_id(
                parent_ids[mutang_indices], points
            )
            lineage_parent_matches += sum(parent is not None for parent in matched_parents)
        else:
            matched_parents = _match_sorbes_parents(
                copied_latents, points, point_latents, match_decimals
            )
            latent_parent_matches += sum(parent is not None for parent in matched_parents)

        nested_progress.start_stage(run.run_id, len(sequences))
        true_latents = encode_sequences(
            autoencoder, list(sequences), encode_batch_size, nested_progress
        )

        for sequence, true_latent, copied_latent, parent in zip(
            sequences, true_latents, copied_latents, matched_parents
        ):
            # The candidate was generated at its parent SORBES point's
            # iteration -- both candidate rows share that trajectory_step.
            generating_step = (
                int(parent["trajectory_step"]) if parent is not None else None
            )
            rows.append(
                {
                    "run_id": run.run_id,
                    "grid_value": grid_value,
                    "sorbes_moved": sorbes_moved,
                    "distance_kind": "candidate_to_origin",
                    "trajectory_step": generating_step,
                    "levenshtein_distance": _levenshtein_distance(
                        sequence, origin_sequence
                    ),
                    "euclidean_distance": float(
                        np.linalg.norm(true_latent - origin_latent)
                    ),
                    "sequence": sequence,
                }
            )
            if parent is None:
                unmatched_parents += 1
                continue
            rows.append(
                {
                    "run_id": run.run_id,
                    "grid_value": grid_value,
                    "sorbes_moved": sorbes_moved,
                    "distance_kind": "candidate_to_sorbes_parent",
                    "trajectory_step": generating_step,
                    "levenshtein_distance": _levenshtein_distance(
                        sequence, str(parent["sequence"])
                    ),
                    "euclidean_distance": float(
                        np.linalg.norm(true_latent - copied_latent)
                    ),
                    "sequence": sequence,
                }
            )

    data = pd.DataFrame(rows)
    if data.empty:
        return AnalysisResult(data, empty_metadata)
    data = data[data["levenshtein_distance"].isin(levenshtein_bins)].reset_index(
        drop=True
    )
    outliers, frequency = compute_outliers(
        data, outlier_iqr_multiplier=outlier_iqr_multiplier
    )

    return AnalysisResult(
        data,
        {
            "analysis": "latent_jump",
            "run_count": len(selection.runs),
            "total_walk_points": total_walk_points,
            "total_mutang_candidates_analyzed": total_candidates_analyzed,
            "unmatched_sorbes_parents": unmatched_parents,
            "lineage_parent_matches": lineage_parent_matches,
            "latent_parent_matches": latent_parent_matches,
            "outlier_iqr_multiplier": outlier_iqr_multiplier,
            "candidate_sample_size": candidate_sample_size,
            "autoencoder": autoencoder_parameters,
        },
        {"outliers": outliers, "outlier_frequency": frequency},
    )
