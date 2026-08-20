"""Latent-space locality of HydrAMP training peptide groups.

Preliminary diagnostic: can a within-group notion of "local neighborhood"
even be defined in HydrAMP's latent space for the peptide groups it was
trained on? Two measures per group:

- distance to each member's nearest neighbors within the same group
  (local density);
- distance from each member to the group's centroid (overall spread).

No BLOSUM, no biological function classes, no significance testing yet --
those come later, once this basic diagnostic has been looked at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pep_compass.analysis.analysis_types._encoding import encode_sequences
from pep_compass.analysis.analysis_types.locality._streaming import NestedProgress
from pep_compass.analysis.result import AnalysisResult
from pep_compass.autoencoder.strategies.hydramp.adapter import HydrampAutoencoder


def load_single_label_groups(
    peptides_path: str | Path,
    *,
    group_column: str = "dataset",
    sequence_column: str = "sequence",
) -> pd.DataFrame:
    """Load peptides whose group label is unambiguous.

    ``peptides.csv``'s group column sometimes lists several source datasets
    for one peptide, joined with ``;`` (about 3% of rows). Those are dropped
    here so every remaining peptide belongs to exactly one group.

    :param peptides_path: Path to the peptides CSV.
    :param group_column: Column naming each peptide's training-data group.
    :param sequence_column: Column holding the peptide sequence.
    :return: ``sequence``/``group`` columns, one row per unambiguous peptide,
        deduplicated by sequence.
    """
    data = pd.read_csv(peptides_path)
    single_label = ~data[group_column].astype(str).str.contains(";")
    data = data.loc[single_label, [sequence_column, group_column]].dropna()
    data = data.rename(columns={sequence_column: "sequence", group_column: "group"})
    return data.drop_duplicates("sequence").reset_index(drop=True)


def load_amp_mic_groups(
    peptides_path: str | Path,
    *,
    group_column: str = "dataset",
    sequence_column: str = "sequence",
) -> pd.DataFrame:
    """Load peptides labeled by HydrAMP's own (c_AMP, c_MIC) training condition.

    HydrAMP conditions generation on ``c = (c_AMP, c_MIC)``: whether a peptide
    is an antimicrobial peptide, and whether it has high activity (MIC) against
    *E. coli*. Only three combinations are biologically coherent (confirmed
    against the paper's own Methods section and its public dataset-preparation
    notebook, ``szczurek-lab/hydramp``):

    - ``AMP_high_MIC`` (c=1,1): confirmed AMP, confirmed MIC <= 32 ug/mL vs.
      *E. coli* -- ``dataset`` tag contains ``hydramp_dbaasp_high_ecoli_activity``.
    - ``AMP_low_MIC`` (c=1,0): confirmed AMP, confirmed MIC > 32 ug/mL vs.
      *E. coli* -- tag contains ``hydramp_dbaasp_low_ecoli_activity``.
    - ``non_AMP`` (c=0,0): confirmed non-AMP background -- tag is exactly one
      of the UniProt splits or the Veltri negative set.

    (c=0,1) is not biologically coherent (MIC only means anything for a
    confirmed AMP) and is never produced. Peptides whose tags don't resolve
    one of the three groups -- no *E. coli* MIC threshold recorded (only
    ``hydramp_mic_data``/``hydramp_dbaasp_clean``/``hydramp_veltri_positive``
    alone), *S. aureus*-only activity tags (paper's MIC condition is
    *E. coli*-specific), or ``ampsphere_amp`` (a much larger, separately
    sourced AMP reference set that does not match the paper's own
    11,131-peptide AMP-positive count) -- are dropped rather than guessed.

    :param peptides_path: Path to the peptides CSV.
    :param group_column: Column holding the ``;``-joined dataset membership tags.
    :param sequence_column: Column holding the peptide sequence.
    :return: ``sequence``/``group`` columns, deduplicated by sequence.
    """
    data = pd.read_csv(peptides_path)
    tags = data[group_column].astype(str)
    group = pd.Series(pd.NA, index=data.index, dtype="object")
    group[tags.str.contains("hydramp_dbaasp_high_ecoli_activity")] = "AMP_high_MIC"
    group[tags.str.contains("hydramp_dbaasp_low_ecoli_activity")] = "AMP_low_MIC"
    group[
        tags.isin(
            [
                "hydramp_uniprot_0_25_train",
                "hydramp_uniprot_0_25_val",
                "hydramp_uniprot_0_25_test",
                "hydramp_veltri_negative",
            ]
        )
    ] = "non_AMP"
    data = data.assign(group=group)[[sequence_column, "group"]].dropna()
    data = data.rename(columns={sequence_column: "sequence"})
    return data.drop_duplicates("sequence").reset_index(drop=True)


def _build_autoencoder(
    *,
    model_name: str,
    device: str,
    jacobian_mode: str,
    jacobian_eps: float,
    field_eps: float,
) -> HydrampAutoencoder:
    """Construct a standalone, ``eval()``-mode encoder for group_locality."""
    autoencoder = HydrampAutoencoder(
        device=device,
        jacobian_mode=jacobian_mode,
        jacobian_eps=jacobian_eps,
        field_eps=field_eps,
        model_name=model_name,
    )
    return autoencoder.to(device).eval()


def _validate_locality_inputs(
    peptides: pd.DataFrame,
    *,
    neighbor_count: int,
    neighbor_sample_size: int,
    neighbor_query_batch: int,
    encode_batch_size: int,
) -> None:
    """Validate shared group-locality inputs before model construction.

    :param peptides: Input frame containing sequence and group columns.
    :param neighbor_count: Neighbors retained per sampled sequence.
    :param neighbor_sample_size: Maximum sequences sampled per group.
    :param neighbor_query_batch: Queries evaluated in one distance chunk.
    :param encode_batch_size: Sequences encoded in one model forward pass.
    :raises ValueError: If required columns are absent or a size is not positive.
    """
    required_columns = {"sequence", "group"}
    missing_columns = required_columns - set(peptides.columns)
    if missing_columns:
        raise ValueError(
            f"Peptide groups are missing required columns: {sorted(missing_columns)}."
        )
    for name, value in {
        "neighbor_count": neighbor_count,
        "neighbor_sample_size": neighbor_sample_size,
        "neighbor_query_batch": neighbor_query_batch,
        "encode_batch_size": encode_batch_size,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive.")


def _encode_groups(
    peptides: pd.DataFrame,
    autoencoder: HydrampAutoencoder,
    encode_batch_size: int,
    nested_progress: NestedProgress,
) -> dict[str, torch.Tensor]:
    """Encode every peptide once per group. Shared by within/between-group functions."""
    latents_by_group: dict[str, torch.Tensor] = {}
    for group_name, group_frame in peptides.groupby("group"):
        sequences = group_frame["sequence"].tolist()
        nested_progress.start_stage(str(group_name), len(sequences))
        latents = encode_sequences(
            autoencoder, sequences, encode_batch_size, nested_progress
        )
        latents_by_group[str(group_name)] = torch.as_tensor(
            latents, device=autoencoder.device
        )
    return latents_by_group


def group_locality(
    peptides: pd.DataFrame,
    *,
    model_name: str = "article_25",
    device: str = "cuda",
    jacobian_mode: str = "approx",
    jacobian_eps: float = 0.001,
    field_eps: float = 0.001,
    neighbor_count: int = 100,
    neighbor_sample_size: int = 5000,
    neighbor_query_batch: int = 512,
    encode_batch_size: int = 1024,
    random_seed: int = 0,
    progress: bool = True,
) -> AnalysisResult:
    """Measure within-group latent locality: nearest-neighbor and centroid distance.

    For every group in ``peptides["group"]``, encodes every member once
    (``z = mu_phi(s)``), then computes:

    - ``neighbor_distance`` (``data``): for a random sample of up to
      ``neighbor_sample_size`` members, the Euclidean distance to each of its
      ``neighbor_count`` nearest neighbors *within the same group*. Exact
      distances, not approximate -- but only for the sampled members, since
      computing this for every member of an ~180k-member group would mean
      ~180k x 180k distances.
    - ``centroid_distance`` (``diagnostics``): for every member (no
      subsampling -- this is O(n), not O(n^2)), the Euclidean distance to the
      group's centroid (mean latent position over the whole group).

    :param peptides: ``sequence``/``group`` columns, e.g. from
        :func:`load_single_label_groups`.
    :param model_name: HydrAMP model name (``autoencoder.model`` in an
        experiment config).
    :param device: Torch device used for encoding and distance computation.
        Defaults to GPU: encoding + distances for the full 302k-peptide
        corpus (8 groups, k=100) takes ~13s on GPU vs. several minutes on
        CPU. Tensors here are small (largest group is 180k x 64 floats,
        largest per-chunk distance matrix ~370MB) -- unlike the MUTANG
        candidate-explosion OOM earlier this session, there is no risk of
        runaway allocation.
    :param jacobian_mode: Required ``HydrampAutoencoder`` constructor
        argument; unused here since no Jacobian is computed.
    :param jacobian_eps: Required ``HydrampAutoencoder`` constructor argument.
    :param field_eps: Required ``HydrampAutoencoder`` constructor argument.
    :param neighbor_count: Nearest neighbors computed per sampled member.
    :param neighbor_sample_size: Members sampled per group for the
        neighbor-distance computation.
    :param neighbor_query_batch: Sampled members processed per distance-matrix
        chunk (bounds peak memory: ``neighbor_query_batch x group_size``).
    :param encode_batch_size: Sequences encoded per autoencoder forward pass.
    :param random_seed: Seed for per-group member subsampling.
    :param progress: Print an overall bar plus one bar per group being encoded.
    :return: ``data`` holds ``neighbor_distance`` rows (one per sampled
        member per neighbor); ``diagnostics["centroid_distance"]`` holds one
        row per member.
    """
    _validate_locality_inputs(
        peptides,
        neighbor_count=neighbor_count,
        neighbor_sample_size=neighbor_sample_size,
        neighbor_query_batch=neighbor_query_batch,
        encode_batch_size=encode_batch_size,
    )
    autoencoder = _build_autoencoder(
        model_name=model_name,
        device=device,
        jacobian_mode=jacobian_mode,
        jacobian_eps=jacobian_eps,
        field_eps=field_eps,
    )
    rng = np.random.default_rng(random_seed)
    groups = peptides.groupby("group")
    nested_progress = NestedProgress(
        "group_locality:encode", len(peptides), enabled=progress
    )

    neighbor_frames: list[pd.DataFrame] = []
    centroid_frames: list[pd.DataFrame] = []
    group_sizes: dict[str, int] = {}

    for group_name, group_frame in groups:
        sequences = group_frame["sequence"].tolist()
        group_sizes[str(group_name)] = len(sequences)
        nested_progress.start_stage(str(group_name), len(sequences))
        latents = encode_sequences(
            autoencoder, sequences, encode_batch_size, nested_progress
        )
        latents_tensor = torch.as_tensor(latents, device=device)

        centroid = latents_tensor.mean(dim=0)
        centroid_distances = torch.linalg.norm(
            latents_tensor - centroid, dim=-1
        ).cpu().numpy()
        centroid_frames.append(
            pd.DataFrame(
                {
                    "group": group_name,
                    "sequence": sequences,
                    "centroid_distance": centroid_distances,
                }
            )
        )

        if len(sequences) < 2:
            continue
        sample_size = min(neighbor_sample_size, len(sequences))
        sample_indices = rng.choice(len(sequences), size=sample_size, replace=False)
        k = min(neighbor_count, len(sequences) - 1)
        sequences_array = np.asarray(sequences)
        for start in range(0, len(sample_indices), neighbor_query_batch):
            chunk_indices = sample_indices[start : start + neighbor_query_batch]
            distances = torch.cdist(
                latents_tensor[chunk_indices], latents_tensor
            )  # (chunk, group_size)
            distances[np.arange(len(chunk_indices)), chunk_indices] = float("inf")
            nearest = torch.topk(distances, k=k, largest=False).values  # (chunk, k)
            neighbor_frames.append(
                pd.DataFrame(
                    {
                        "group": group_name,
                        "sequence": np.repeat(sequences_array[chunk_indices], k),
                        "neighbor_distance": nearest.cpu().numpy().reshape(-1),
                    }
                )
            )

    neighbor_data = (
        pd.concat(neighbor_frames, ignore_index=True)
        if neighbor_frames
        else pd.DataFrame(columns=["group", "sequence", "neighbor_distance"])
    )
    centroid_data = (
        pd.concat(centroid_frames, ignore_index=True)
        if centroid_frames
        else pd.DataFrame(columns=["group", "sequence", "centroid_distance"])
    )
    return AnalysisResult(
        neighbor_data,
        {
            "analysis": "group_locality",
            "group_count": len(group_sizes),
            "group_sizes": group_sizes,
            "neighbor_count": neighbor_count,
            "neighbor_sample_size": neighbor_sample_size,
            "model_name": model_name,
        },
        {"centroid_distance": centroid_data},
    )


def between_group_distance(
    peptides: pd.DataFrame,
    *,
    model_name: str = "article_25",
    device: str = "cuda",
    jacobian_mode: str = "approx",
    jacobian_eps: float = 0.001,
    field_eps: float = 0.001,
    neighbor_count: int = 100,
    neighbor_sample_size: int = 5000,
    neighbor_query_batch: int = 512,
    encode_batch_size: int = 1024,
    random_seed: int = 0,
    progress: bool = True,
) -> AnalysisResult:
    """Measure between-group latent separation: nearest-neighbor and centroid distance.

    Directly comparable to :func:`group_locality`'s within-group numbers --
    same metric, same ``neighbor_count``, same ``neighbor_sample_size`` -- so
    "within" and "between" can be read side by side: if between-group
    distance is no larger than within-group distance, the groups are not
    actually separated in latent space, whatever their within-group numbers
    looked like on their own.

    For every ORDERED pair of distinct groups ``(a, b)``:

    - ``neighbor_distance`` (``data``): a random sample of up to
      ``neighbor_sample_size`` members of ``a``, each compared against EVERY
      member of ``b``, keeping the ``neighbor_count`` smallest distances --
      "how far is a typical member of a from its nearest neighbors in b".
    - ``centroid_distance`` (``diagnostics``): exact centroid-to-centroid
      distance, using ALL members of both groups (centroids already
      summarize the full population -- no sampling needed for this part).

    :param peptides: ``sequence``/``group`` columns, e.g. from
        :func:`load_single_label_groups`.
    :param model_name: HydrAMP model name.
    :param device: Torch device for encoding and distance computation.
    :param jacobian_mode: Required ``HydrampAutoencoder`` constructor argument.
    :param jacobian_eps: Required ``HydrampAutoencoder`` constructor argument.
    :param field_eps: Required ``HydrampAutoencoder`` constructor argument.
    :param neighbor_count: Nearest neighbors computed per sampled member.
    :param neighbor_sample_size: Members of ``a`` sampled per group pair.
    :param neighbor_query_batch: Sampled members processed per distance-matrix
        chunk.
    :param encode_batch_size: Sequences encoded per autoencoder forward pass.
    :param random_seed: Seed for per-group member subsampling.
    :param progress: Print an overall bar plus one bar per group being encoded.
    :return: ``data`` holds between-group ``neighbor_distance`` rows, each
        tagged with ``group`` (the sampled member's own group) and
        ``other_group`` (the group its neighbors were found in);
        ``diagnostics["centroid_distance"]`` holds one row per ordered pair.
    """
    _validate_locality_inputs(
        peptides,
        neighbor_count=neighbor_count,
        neighbor_sample_size=neighbor_sample_size,
        neighbor_query_batch=neighbor_query_batch,
        encode_batch_size=encode_batch_size,
    )
    autoencoder = _build_autoencoder(
        model_name=model_name,
        device=device,
        jacobian_mode=jacobian_mode,
        jacobian_eps=jacobian_eps,
        field_eps=field_eps,
    )
    rng = np.random.default_rng(random_seed)
    group_names = sorted(peptides["group"].unique())
    nested_progress = NestedProgress(
        "between_group_distance:encode", len(peptides), enabled=progress
    )
    latents_by_group = _encode_groups(
        peptides, autoencoder, encode_batch_size, nested_progress
    )
    sequences_by_group = {
        str(name): frame["sequence"].tolist()
        for name, frame in peptides.groupby("group")
    }
    centroids = {
        name: latents.mean(dim=0) for name, latents in latents_by_group.items()
    }

    neighbor_frames: list[pd.DataFrame] = []
    centroid_rows: list[dict[str, Any]] = []

    for group_a in group_names:
        sequences_a = sequences_by_group[group_a]
        sequences_a_array = np.asarray(sequences_a)
        latents_a = latents_by_group[group_a]
        sample_size = min(neighbor_sample_size, len(sequences_a))
        sample_indices = rng.choice(len(sequences_a), size=sample_size, replace=False)

        for group_b in group_names:
            if group_a == group_b:
                continue
            centroid_rows.append(
                {
                    "group": group_a,
                    "other_group": group_b,
                    "centroid_distance": torch.linalg.norm(
                        centroids[group_a] - centroids[group_b]
                    ).item(),
                }
            )

            latents_b = latents_by_group[group_b]
            k = min(neighbor_count, len(sequences_by_group[group_b]))
            for start in range(0, len(sample_indices), neighbor_query_batch):
                chunk_indices = sample_indices[start : start + neighbor_query_batch]
                distances = torch.cdist(
                    latents_a[chunk_indices], latents_b
                )  # (chunk, |b|)
                nearest = torch.topk(distances, k=k, largest=False).values  # (chunk, k)
                neighbor_frames.append(
                    pd.DataFrame(
                        {
                            "group": group_a,
                            "other_group": group_b,
                            "sequence": np.repeat(sequences_a_array[chunk_indices], k),
                            "neighbor_distance": nearest.cpu().numpy().reshape(-1),
                        }
                    )
                )

    neighbor_data = (
        pd.concat(neighbor_frames, ignore_index=True)
        if neighbor_frames
        else pd.DataFrame(columns=["group", "other_group", "sequence", "neighbor_distance"])
    )
    centroid_data = pd.DataFrame(centroid_rows)
    return AnalysisResult(
        neighbor_data,
        {
            "analysis": "between_group_distance",
            "group_count": len(group_names),
            "pair_count": len(group_names) * (len(group_names) - 1),
            "neighbor_count": neighbor_count,
            "neighbor_sample_size": neighbor_sample_size,
            "model_name": model_name,
        },
        {"centroid_distance": centroid_data},
    )
