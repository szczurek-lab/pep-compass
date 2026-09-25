r"""Diagnostics for MEROPS specificity matrices.

``cleavage.py`` consumes MEROPS specificity matrices as a fixed environment. This
module answers the prior question: **how much specificity information does a
matrix actually carry**, given that MEROPS pools a highly uneven number of
recorded cleavages per peptidase (from a single cleavage to several thousand).

Most diagnostics are matrix-level. The window null additionally takes an
empirical peptide-length sample:

* :func:`positional_information_content` -- per-subsite divergence from a
  reference residue distribution,
* :func:`expected_null_information_content` -- the divergence a subsite shows
  when its residues are drawn from that same reference, i.e. the finite-sample
  floor,
* :func:`information_content_excess` -- the difference of the two,
* :func:`simulate_window_null_divergence` -- divergence from complete null
  cleavage datasets at specified event counts,
* :func:`bootstrap_bond_reduction_rank_stability` -- ranking agreement under
  four reductions of masked per-bond log-odds,
* :func:`bootstrap_specificity_matrices` -- resampled matrices for propagating
  cleavage-count uncertainty into a downstream score,
* :func:`profile_similarity_matrix` / :func:`cluster_profiles` -- how many
  distinct specificity profiles a panel holds.

Every metric takes the reference distribution :math:`q` as an argument. The
choice of :math:`q` determines what the metric measures: against a uniform
:math:`q_a = 1/20` a subsite that only reflects the composition of the substrate
pool is reported as specific, because residue frequencies in real proteomes span
roughly an order of magnitude. Background distributions are constructed in
:mod:`~pep_compass.optimization.components.helpers.proteolysis.background`.

Dimension symbols used throughout: ``N`` proteases, ``P`` subsites (8, ``P4``
… ``P4'``), ``A`` amino acids (20), ``B`` simulation or bootstrap replicates.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

from pep_compass.optimization.components.helpers.proteolysis.background import (
    normalize_background,
)
from pep_compass.optimization.components.helpers.proteolysis.cleavage import (
    DEFAULT_MEROPS_DATASETS,
    MEROPS_AMINO_ACIDS,
    MEROPS_SUBSITE_OFFSETS,
    _default_merops_root,
)

# Smoothing prior behind ``matrices_logprob.npy``.
# REMARK: Verified numerically against data/merops/35_human_serum: the shipped
# log-probabilities equal log((counts + 0.5) / (n + 0.5 * 20)), i.e. a Jeffreys
# prior. Resampling must reuse the same constant, otherwise bootstrap replicates
# are not comparable with the distributed matrices.
MEROPS_SMOOTHING_ALPHA = 0.5

N_AMINO_ACIDS = len(MEROPS_AMINO_ACIDS)
BOND_REDUCTIONS = ("max", "sum", "mean", "logsumexp")


def load_panel_metadata(
    dataset_ids: int | Iterable[int] = DEFAULT_MEROPS_DATASETS,
    merops_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load the per-peptidase metadata of one or more MEROPS datasets.

    ``load_protease_panel`` returns matrices and codes only. This function
    returns the remaining ``codes.json`` fields (display name, pooled cleavage
    count, serum category) in panel row order, so diagnostics can be reported
    per named peptidase.

    :param dataset_ids: Integer dataset id or ids, in the same convention as
        :func:`~pep_compass.optimization.components.helpers.proteolysis.cleavage.load_protease_panel`.
    :param merops_root: Directory holding ``index.json`` and the dataset folders.
        Defaults to the repository ``data/merops``.
    :return: One dictionary per panel row, carrying at least ``row``, ``code``,
        ``name``, ``n_cleavages`` and the originating ``dataset_id``.
    :raises FileNotFoundError: If the MEROPS index or a dataset folder is absent.
    """

    if isinstance(dataset_ids, int):
        dataset_ids = [dataset_ids]
    root = Path(merops_root) if merops_root is not None else _default_merops_root()
    index = json.loads((root / "index.json").read_text())

    entries: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        folder = root / index["by_index"][str(dataset_id)]["folder"]
        metadata = json.loads((folder / "codes.json").read_text())
        for peptidase in sorted(metadata["peptidases"], key=lambda item: item["row"]):
            entries.append({**peptidase, "dataset_id": int(dataset_id)})
    return entries


def merops_code_locations(
    merops_root: str | Path | None = None,
) -> dict[str, dict[int, int]]:
    """Index every peptidase code by the datasets that hold a matrix for it.

    A MEROPS code can appear in several datasets, with a different number of
    pooled cleavages in each, and a code named by an experiment is not
    necessarily in the dataset an analysis happens to be using. This index makes
    the lookup explicit instead of assuming one panel.

    :param merops_root: Directory holding ``index.json`` and the dataset folders.
        Defaults to the repository ``data/merops``.
    :return: Mapping from peptidase code to ``{dataset_id: n_cleavages}``.
    :raises FileNotFoundError: If the MEROPS index is absent.
    """

    root = Path(merops_root) if merops_root is not None else _default_merops_root()
    index = json.loads((root / "index.json").read_text())

    locations: dict[str, dict[int, int]] = {}
    for key, entry in index["by_index"].items():
        metadata = json.loads((root / entry["folder"] / "codes.json").read_text())
        for peptidase in metadata["peptidases"]:
            locations.setdefault(peptidase["code"], {})[int(key)] = int(peptidase["n_cleavages"])
    return locations


def best_merops_dataset(
    code: str,
    merops_root: str | Path | None = None,
    locations: dict[str, dict[int, int]] | None = None,
) -> tuple[int | None, int]:
    """Return the dataset holding the most cleavage evidence for one code.

    :param code: MEROPS peptidase code.
    :param merops_root: Directory holding the MEROPS datasets.
    :param locations: Precomputed output of :func:`merops_code_locations`, to
        avoid rereading the index for every code.
    :return: Tuple of dataset id and cleavage count, ``(None, 0)`` when the code
        has no matrix in any dataset.
    """

    if locations is None:
        locations = merops_code_locations(merops_root)
    entry = locations.get(code)
    if not entry:
        return None, 0
    dataset_id = max(entry, key=lambda key: entry[key])
    return dataset_id, entry[dataset_id]


def _smoothed_frequencies(counts: np.ndarray, alpha: float = MEROPS_SMOOTHING_ALPHA) -> np.ndarray:
    """Convert cleavage counts into smoothed residue frequencies.

    :param counts: Cleavage counts, shape ``(..., A)``.
    :param alpha: Dirichlet smoothing prior per residue.
    :return: Residue frequencies summing to one along the last axis, ``(..., A)``.
    """

    smoothed = counts.astype(np.float64) + alpha  # (..., A)
    return smoothed / smoothed.sum(axis=-1, keepdims=True)  # (..., A)


def positional_information_content(
    counts: np.ndarray,
    alpha: float = MEROPS_SMOOTHING_ALPHA,
    background: np.ndarray | None = None,
) -> np.ndarray:
    r"""Compute the divergence of each subsite from a reference distribution.

    The metric is the Kullback-Leibler divergence in bits,

    .. math::
        \mathrm{IC}(p \| q) = \sum_a p_a \log_2 \frac{p_a}{q_a},

    where :math:`p` is the smoothed residue distribution of the subsite and
    :math:`q` the reference. It is zero when the subsite matches the reference
    and grows as it departs from it, in either direction.

    A uniform :math:`q` reduces this to :math:`\log_2 A - H(p)`, the classical
    sequence-logo information content. That form treats agreement with the
    composition of the substrate pool as specificity, so a proteome background
    should be supplied whenever the substrates were drawn from a proteome.

    :param counts: Cleavage counts, shape ``(N, P, A)`` or ``(P, A)``.
    :param alpha: Dirichlet smoothing prior, matching the distributed matrices.
    :param background: Reference residue distribution ``(A,)``. Defaults to
        uniform.
    :return: Divergence in bits, shape ``(N, P)`` or ``(P,)``.
    """

    frequencies = _smoothed_frequencies(counts, alpha)  # (..., A)
    reference = normalize_background(background, frequencies.shape[-1])  # (A,)
    return (frequencies * np.log2(frequencies / reference)).sum(axis=-1)  # (...)


def expected_null_information_content(
    n_observations: np.ndarray | int,
    alpha: float = MEROPS_SMOOTHING_ALPHA,
    n_samples: int = 400,
    seed: int = 0,
    n_amino_acids: int = N_AMINO_ACIDS,
    background: np.ndarray | None = None,
) -> np.ndarray:
    r"""Estimate the divergence produced by sampling noise alone.

    A subsite observed a handful of times diverges from the reference even when
    its residues were drawn from that reference, because a small multinomial
    sample cannot reproduce a 20-residue distribution. This function measures
    that floor by Monte Carlo: residues are drawn from the reference,

    .. math::
        N^{(b)} \sim \mathrm{Multinomial}(n, q),

    and the divergence of each draw from the same :math:`q` is averaged. The
    null and the observed statistic therefore use one reference distribution;
    passing different ones makes the difference uninterpretable.

    :param n_observations: Number of observed cleavages per subsite, any shape.
    :param alpha: Dirichlet smoothing prior, matching the observed matrices.
    :param n_samples: Monte Carlo replicates per distinct observation count.
    :param seed: Seed of the Monte Carlo generator.
    :param n_amino_acids: Size of the residue alphabet.
    :param background: Reference residue distribution ``(A,)`` the null draws
        from. Defaults to uniform.
    :return: Expected divergence in bits, shaped like ``n_observations``.

    .. note::
        The floor is **not** monotonically decreasing in the number of
        observations. The Jeffreys prior of the distributed matrices pulls a very
        small sample towards a uniform distribution, so for a non-uniform
        reference the floor first falls, reaches a minimum where the prior and
        the data balance, and then rises again before decaying. The excess over
        the floor must therefore not be read as a ranking of data quality across
        very different sample sizes.
    """

    observations = np.asarray(n_observations)
    generator = np.random.default_rng(seed)
    reference = normalize_background(background, n_amino_acids)  # (A,)

    ## REMARK: The floor depends only on the number of draws, and MEROPS counts
    ## repeat heavily across the panel, so each distinct count is simulated once.
    expectations: dict[int, float] = {}
    for count in np.unique(observations.astype(int)):
        if count <= 0:
            expectations[int(count)] = 0.0
            continue
        samples = generator.multinomial(int(count), reference, size=n_samples)  # (n_samples, A)
        expectations[int(count)] = float(
            positional_information_content(samples, alpha, background=reference).mean()
        )

    lookup = np.vectorize(lambda value: expectations[int(value)], otypes=[np.float64])
    return lookup(observations)


def information_content_excess(
    counts: np.ndarray,
    alpha: float = MEROPS_SMOOTHING_ALPHA,
    n_samples: int = 400,
    seed: int = 0,
    background: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the subsite divergence into a sampling floor and an excess.

    The reference distribution is passed to both terms, so the excess is the part
    of the departure from the reference that drawing from that reference does not
    already produce.

    :param counts: Cleavage counts, shape ``(N, P, A)``.
    :param alpha: Dirichlet smoothing prior.
    :param n_samples: Monte Carlo replicates per distinct observation count.
    :param seed: Seed of the Monte Carlo generator.
    :param background: Reference residue distribution ``(A,)``. Defaults to
        uniform.
    :return: Tuple of observed divergence ``(N, P)``, expected floor ``(N, P)``
        and their difference ``(N, P)``.
    """

    observed = positional_information_content(counts, alpha, background=background)  # (N, P)
    n_observations = counts.sum(axis=-1)  # (N, P)
    null = expected_null_information_content(
        n_observations,
        alpha=alpha,
        n_samples=n_samples,
        seed=seed,
        n_amino_acids=counts.shape[-1],
        background=background,
    )  # (N, P)
    return observed, null, observed - null


def simulate_window_null_divergence(
    sample_sizes: np.ndarray,
    peptide_lengths: np.ndarray,
    *,
    n_replicates: int = 600,
    seed: int = 0,
    background: np.ndarray | None = None,
    alpha: float = MEROPS_SMOOTHING_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Simulate null specificity from peptide lengths and cleavage windows.

    Each replicate at sample size ``n`` contains exactly ``n`` cleavage events.
    An event draws a peptide length from the empirical length distribution and
    then a bond uniformly among its internal bonds. The bond determines which
    P4--P4' positions exist. Residues at existing positions are independent
    draws from ``background``; equivalent multinomial draws are batched by
    subsite after the window coverage has been counted. The returned divergence
    is the mean of the eight smoothed subsite divergences, including the prior-
    only estimate at an unobserved subsite, matching the observed panel metric.

    :param sample_sizes: Positive numbers of cleavage events, shape ``(G,)``.
    :param peptide_lengths: Empirical peptide lengths, each at least two residues.
    :param n_replicates: Independent null datasets per sample size.
    :param seed: Random seed for lengths, bonds, and residues.
    :param background: Residue probabilities, shape ``(A,)``.
    :param alpha: Per-residue smoothing prior used for the observed matrices.
    :return: Mean divergences ``(G, B)`` and subsite depths ``(G, B, 8)``.
    :raises ValueError: If sample sizes, lengths, or replicate count are invalid.

    .. note::
        Runtime is ``O(B * (sum(sample_sizes) * P + G * P * A))`` and peak
        auxiliary memory is ``O(B * (max(sample_sizes) + P * A) + G * B * P)`` for ``P=8``
        and ``A`` residue types. Length probabilities are precomputed once.
    """

    sizes = np.asarray(sample_sizes)
    lengths = np.asarray(peptide_lengths)
    if (
        sizes.ndim != 1
        or sizes.size == 0
        or not np.all(np.isfinite(sizes))
        or np.any(sizes < 1)
        or np.any(sizes != np.floor(sizes))
    ):
        raise ValueError("sample_sizes must be a nonempty array of positive integers.")
    if (
        lengths.ndim != 1
        or lengths.size == 0
        or not np.all(np.isfinite(lengths))
        or np.any(lengths < 2)
        or np.any(lengths != np.floor(lengths))
    ):
        raise ValueError("peptide_lengths must be a nonempty array of integers >= 2.")
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive.")

    reference = normalize_background(background, N_AMINO_ACIDS)  # (A,)
    length_values, length_counts = np.unique(lengths.astype(np.int64), return_counts=True)  # (L,), (L,)
    length_probabilities = length_counts / length_counts.sum()  # (L,)
    generator = np.random.default_rng(seed)
    divergences = np.empty((len(sizes), n_replicates), dtype=np.float64)  # (G, B)
    depths = np.empty((len(sizes), n_replicates, 8), dtype=np.int64)  # (G, B, P)

    # Generate exactly n candidate cleavage windows for each replicate.
    for grid_index, sample_size in enumerate(sizes.astype(np.int64)):
        sampled_lengths = generator.choice(
            length_values, size=(n_replicates, int(sample_size)), p=length_probabilities
        )  # (B, n)
        bonds = 1 + np.floor(
            generator.random(sampled_lengths.shape) * (sampled_lengths - 1)
        ).astype(np.int64)  # (B, n); bond is between residues bond-1 and bond

        ## Count which positions of each generated window are within its peptide.
        for subsite, offset in enumerate(range(-4, 4)):
            residue_index = bonds + offset  # (B, n)
            depths[grid_index, :, subsite] = (
                (residue_index >= 0) & (residue_index < sampled_lengths)
            ).sum(axis=1)  # (B,)

        ## Draw background residues at the observed positions and score them.
        residue_counts = generator.multinomial(
            depths[grid_index], reference
        )  # (B, P, A)
        subsite_divergence = positional_information_content(
            residue_counts, alpha=alpha, background=reference
        )  # (B, P)
        divergences[grid_index] = subsite_divergence.mean(axis=1)  # (B,)

    return divergences, depths


def _peptide_window_residue_columns(sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Index observed residues at every internal bond of each peptide.

    :param sequences: Peptides over the MEROPS amino-acid alphabet.
    :return: Residue columns ``(I, J, P)`` with ``-1`` for missing positions or
        padded bonds, and valid-bond mask ``(I, J)``.
    :raises ValueError: If a peptide is too short or has an invalid residue.
    """

    if not sequences or any(len(sequence) < 2 for sequence in sequences):
        raise ValueError("sequences must contain peptides of at least two residues.")
    alphabet = {residue: column for column, residue in enumerate(MEROPS_AMINO_ACIDS)}
    n_bonds = max(map(len, sequences)) - 1
    columns = np.full((len(sequences), n_bonds, len(MEROPS_SUBSITE_OFFSETS)), -1, dtype=np.int16)  # (I, J, P)
    bond_valid = np.zeros((len(sequences), n_bonds), dtype=bool)  # (I, J)
    offsets = np.asarray(MEROPS_SUBSITE_OFFSETS)  # (P,)

    # Encode each peptide once; all bootstrap matrices use these same windows.
    for peptide_index, sequence in enumerate(sequences):
        try:
            residues = np.asarray([alphabet[residue] for residue in sequence], dtype=np.int16)  # (L,)
        except KeyError as error:
            raise ValueError(f"residue {error.args[0]!r} is outside the MEROPS alphabet.") from error
        bonds = np.arange(len(sequence) - 1)  # (J_i,)
        positions = bonds[:, None] + offsets[None, :]  # (J_i, P)
        observed = (positions >= 0) & (positions < len(sequence))  # (J_i, P)
        columns[peptide_index, : len(bonds)] = np.where(
            observed, residues[np.clip(positions, 0, len(sequence) - 1)], -1
        )  # (J_i, P)
        bond_valid[peptide_index, : len(bonds)] = True
    return columns, bond_valid


def _bond_reduced_log_odds(
    matrices: np.ndarray,
    residue_columns: np.ndarray,
    bond_valid: np.ndarray,
    log_background: np.ndarray,
) -> np.ndarray:
    """Score the same peptide windows under four bond reductions.

    :param matrices: Log-probability matrices ``(N, P, A)``.
    :param residue_columns: Prepared window residue columns ``(I, J, P)``.
    :param bond_valid: Valid-bond mask ``(I, J)``.
    :param log_background: Reference log-probabilities ``(A,)``.
    :return: Reduced peptide scores ``(R, I, N)`` in ``BOND_REDUCTIONS`` order.
    """

    n_peptides, n_bonds, _ = residue_columns.shape
    n_proteases = matrices.shape[0]
    log_odds = matrices - log_background[None, None, :]  # (N, P, A)
    bond_scores = np.zeros((n_peptides, n_bonds, n_proteases), dtype=np.float64)  # (I, J, N)

    # Sum log-odds only where the peptide contains the subsite residue.
    for subsite in range(residue_columns.shape[-1]):
        columns = residue_columns[:, :, subsite]  # (I, J)
        sampled = np.take(log_odds[:, subsite], columns.clip(min=0), axis=1)  # (N, I, J)
        bond_scores += np.where(columns[:, :, None] >= 0, sampled.transpose(1, 2, 0), 0.0)

    ## Reduce over valid bonds; padded bonds never enter max or logsumexp.
    masked_scores = np.where(bond_valid[:, :, None], bond_scores, -np.inf)  # (I, J, N)
    score_sum = bond_scores.sum(axis=1)  # (I, N); padded scores are zero
    return np.stack(
        [
            masked_scores.max(axis=1),
            score_sum,
            score_sum / bond_valid.sum(axis=1)[:, None],
            logsumexp(masked_scores, axis=1),
        ],
        axis=0,
    )  # (R, I, N)


def bootstrap_bond_reduction_rank_stability(
    sequences: list[str],
    reference_matrices: np.ndarray,
    bootstrap_matrices: np.ndarray,
    *,
    background: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare peptide rankings across bootstrap matrices for four reductions.

    The local score is the sum of log-probability minus log-background for
    observed P4--P4' residues. ``max``, ``sum``, ``mean``, and ``logsumexp``
    reduce these local scores over every internal bond of a peptide. For each
    protease, Spearman's rho compares the reference peptide scores to those
    from one bootstrap matrix. Undefined rho from constant ranks is set to zero.

    :param sequences: Fixed peptide probe, each length at least two.
    :param reference_matrices: Shipped MEROPS log-probabilities ``(N, P, A)``.
    :param bootstrap_matrices: Resampled log-probabilities ``(B, N, P, A)``.
    :param background: Reference residue frequencies ``(A,)`` for log-odds.
    :return: Reference scores ``(R, I, N)`` and Spearman rho ``(R, B, N)``
        in :data:`BOND_REDUCTIONS` order.
    :raises ValueError: If matrix shapes or probe sequences are invalid.

    .. note::
        With ``I`` peptides, at most ``J`` bonds, ``N`` proteases, ``P``
        subsites, ``A`` residues and ``B`` bootstraps, time is
        ``O(B * I * J * N * P + B * R * I * N log I)``; auxiliary memory is
        ``O(I * J * (P + N) + R * I * N + R * B * N)``. The peptide windows and
        reference ranks are computed once and shared by every bootstrap.
    """

    reference = np.asarray(reference_matrices, dtype=np.float64)
    bootstraps = np.asarray(bootstrap_matrices, dtype=np.float64)
    if (
        reference.ndim != 3
        or reference.shape[1:] != (len(MEROPS_SUBSITE_OFFSETS), N_AMINO_ACIDS)
        or bootstraps.ndim != 4
        or bootstraps.shape[1:] != reference.shape
        or bootstraps.shape[0] < 1
    ):
        raise ValueError("reference and bootstrap matrices must have shapes (N, P, A) and (B, N, P, A).")

    log_background = np.log(normalize_background(background, N_AMINO_ACIDS))  # (A,)
    residue_columns, bond_valid = _peptide_window_residue_columns(sequences)  # (I, J, P), (I, J)
    reference_scores = _bond_reduced_log_odds(
        reference, residue_columns, bond_valid, log_background
    )  # (R, I, N)
    reference_ranks = rankdata(reference_scores, axis=1)  # (R, I, N)
    reference_centered = reference_ranks - reference_ranks.mean(axis=1, keepdims=True)  # (R, I, N)
    reference_norm = np.linalg.norm(reference_centered, axis=1)  # (R, N)
    stability = np.empty((len(BOND_REDUCTIONS), len(bootstraps), len(reference)), dtype=np.float64)  # (R, B, N)

    # Recompute scores under each bootstrap and correlate peptide rank vectors.
    for bootstrap_index, matrices in enumerate(bootstraps):
        scores = _bond_reduced_log_odds(
            matrices, residue_columns, bond_valid, log_background
        )  # (R, I, N)
        ranks = rankdata(scores, axis=1)  # (R, I, N)
        centered = ranks - ranks.mean(axis=1, keepdims=True)  # (R, I, N)
        numerator = np.sum(reference_centered * centered, axis=1)  # (R, N)
        denominator = reference_norm * np.linalg.norm(centered, axis=1)  # (R, N)
        stability[:, bootstrap_index, :] = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
        ).clip(-1.0, 1.0)  # (R, N)

    return reference_scores, stability


def bootstrap_specificity_matrices(
    counts: np.ndarray,
    n_replicates: int = 64,
    alpha: float = MEROPS_SMOOTHING_ALPHA,
    seed: int = 0,
) -> np.ndarray:
    """Resample specificity matrices to propagate cleavage-count uncertainty.

    Each subsite is resampled from its own multinomial with the observed number
    of draws, then converted back to log-probabilities with the same smoothing
    as the distributed matrices.

    :param counts: Cleavage counts, shape ``(N, P, A)``.
    :param n_replicates: Number of bootstrap replicates ``B``.
    :param alpha: Dirichlet smoothing prior, matching the distributed matrices.
    :param seed: Seed of the resampling generator.
    :return: Bootstrap log-probability matrices, shape ``(B, N, P, A)``.

    .. note::
        MEROPS distributes pooled per-subsite counts, not individual cleavage
        records, so subsites are resampled independently. This matches the
        independence assumption the specificity matrices already encode, but it
        cannot reproduce correlations between neighbouring subsites of the same
        cleavage.
    """

    generator = np.random.default_rng(seed)
    observed = counts.astype(np.float64)  # (N, P, A)
    n_observations = observed.sum(axis=-1)  # (N, P)
    frequencies = _smoothed_frequencies(observed, alpha)  # (N, P, A)

    n_proteases, n_subsites, n_residues = observed.shape
    replicates = np.empty((n_replicates, n_proteases, n_subsites, n_residues), dtype=np.float64)

    ## Resample every (protease, subsite) cell with its own observed depth
    ### REMARK: All replicates of one cell are drawn in a single call, which keeps
    ### the number of generator invocations at N * P instead of B * N * P.
    for protease in range(n_proteases):
        for subsite in range(n_subsites):
            draws = int(n_observations[protease, subsite])
            if draws <= 0:
                replicates[:, protease, subsite] = observed[protease, subsite]
                continue
            replicates[:, protease, subsite] = generator.multinomial(
                draws, frequencies[protease, subsite], size=n_replicates
            )  # (B, A)

    return np.log(_smoothed_frequencies(replicates, alpha))  # (B, N, P, A)


def specificity_log_odds(
    log_probabilities: np.ndarray,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Convert specificity log-probabilities into log-odds against a background.

    The log-odds form is what a window score actually sums, so panel redundancy
    should be measured on it rather than on raw log-probabilities.

    :param log_probabilities: Specificity matrices, shape ``(..., P, A)``.
    :param background: Background residue distribution ``(A,)``. Defaults to
        uniform.
    :return: Log-odds matrices of the same shape as ``log_probabilities``.
    """

    reference = normalize_background(background, log_probabilities.shape[-1])  # (A,)
    return log_probabilities - np.log(reference)


def profile_similarity_matrix(
    log_probabilities: np.ndarray,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Correlate the flattened specificity profiles of a protease panel.

    :param log_probabilities: Specificity matrices, shape ``(N, P, A)``.
    :param background: Background residue distribution ``(A,)``, uniform by
        default.
    :return: Pearson correlation matrix of the flattened log-odds, ``(N, N)``.
    """

    log_odds = specificity_log_odds(log_probabilities, background)  # (N, P, A)
    flattened = log_odds.reshape(log_odds.shape[0], -1)  # (N, P * A)
    return np.corrcoef(flattened)  # (N, N)


def cluster_profiles(similarity: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """Group proteases whose specificity profiles are mutually redundant.

    Clusters are formed by average-linkage agglomeration on the correlation
    distance ``1 - r``, cut so that members of one cluster have an average
    correlation of at least ``threshold``.

    :param similarity: Correlation matrix, shape ``(N, N)``.
    :param threshold: Minimum within-cluster correlation.
    :return: Cluster label per protease, shape ``(N,)``.
    :raises ValueError: If ``similarity`` is not square.
    """

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square correlation matrix.")
    if similarity.shape[0] < 2:
        return np.ones(similarity.shape[0], dtype=int)

    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    distance = np.clip((distance + distance.T) / 2.0, 0.0, None)
    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    return fcluster(linkage_matrix, t=1.0 - threshold, criterion="distance")
