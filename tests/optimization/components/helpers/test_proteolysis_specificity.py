"""Tests for the MEROPS specificity-matrix diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from pep_compass.optimization.components.helpers.proteolysis import (
    MEROPS_SMOOTHING_ALPHA,
    best_merops_dataset,
    bootstrap_specificity_matrices,
    cluster_profiles,
    expected_null_information_content,
    information_content_excess,
    merops_code_locations,
    positional_information_content,
    profile_similarity_matrix,
    specificity_log_odds,
)
from pep_compass.optimization.components.helpers.proteolysis.specificity import (
    N_AMINO_ACIDS,
)

# A markedly non-uniform reference, standing in for a proteome composition.
SKEWED_BACKGROUND = np.linspace(1.0, 9.0, N_AMINO_ACIDS)
SKEWED_BACKGROUND = SKEWED_BACKGROUND / SKEWED_BACKGROUND.sum()


N_SUBSITES = 8


def _deterministic_counts(residue_index: int, depth: int) -> np.ndarray:
    """Build counts where every subsite always observes one residue.

    :param residue_index: Column of the always-observed residue.
    :param depth: Number of cleavages per subsite.
    :return: Counts of shape ``(1, P, A)``.
    """

    counts = np.zeros((1, N_SUBSITES, N_AMINO_ACIDS))
    counts[:, :, residue_index] = depth
    return counts


def _uniform_counts(depth_per_residue: int) -> np.ndarray:
    """Build counts spread evenly over all residues.

    :param depth_per_residue: Count assigned to each residue at each subsite.
    :return: Counts of shape ``(1, P, A)``.
    """

    return np.full((1, N_SUBSITES, N_AMINO_ACIDS), float(depth_per_residue))


class TestPositionalInformationContent:
    """Information content of subsite distributions."""

    def test_uniform_counts_carry_no_information(self) -> None:
        """A perfectly uniform subsite has zero information content."""

        content = positional_information_content(_uniform_counts(10))
        assert content.shape == (1, N_SUBSITES)
        assert np.allclose(content, 0.0, atol=1e-12)

    def test_deterministic_counts_approach_the_maximum(self) -> None:
        """A subsite dominated by one residue approaches log2(20) bits."""

        content = positional_information_content(_deterministic_counts(3, 10_000))
        assert np.all(content > np.log2(N_AMINO_ACIDS) - 0.05)

    def test_deeper_sampling_reduces_apparent_information(self) -> None:
        """Identical composition observed more often looks less specific.

        With a fixed residue composition the smoothing prior matters less as the
        depth grows, so a sparsely observed subsite reports inflated content.
        """

        shallow = positional_information_content(_uniform_counts(1))
        deep = positional_information_content(_uniform_counts(100))
        assert np.all(shallow <= deep + 1e-12)

    def test_uniform_reference_reduces_to_the_classical_form(self) -> None:
        """Against a uniform reference the metric is log2(A) minus the entropy.

        This is the sequence-logo information content, and it is the special case
        the default argument produces.
        """

        counts = np.zeros((N_SUBSITES, N_AMINO_ACIDS))
        counts[:, 0] = 3.0
        counts[:, 1] = 1.0
        smoothed = counts + MEROPS_SMOOTHING_ALPHA
        frequencies = smoothed / smoothed.sum(axis=-1, keepdims=True)
        expected = np.log2(N_AMINO_ACIDS) + (frequencies * np.log2(frequencies)).sum(axis=-1)
        assert np.allclose(positional_information_content(counts), expected)


class TestReferenceDistribution:
    """Behaviour of the metric under a non-uniform reference."""

    def test_subsite_matching_the_reference_scores_zero(self) -> None:
        """A subsite that only reflects the reference carries no specificity.

        This is the defect a uniform reference hides: residue frequencies differ
        by roughly an order of magnitude in a real proteome, so a subsite drawn
        from that proteome is reported as specific whenever the reference is
        uniform.
        """

        counts = np.tile(SKEWED_BACKGROUND * 1_000_000, (1, N_SUBSITES, 1))
        against_reference = positional_information_content(counts, background=SKEWED_BACKGROUND)
        against_uniform = positional_information_content(counts)
        assert np.all(np.abs(against_reference) < 1e-3)
        assert np.all(against_uniform > 0.1)

    def test_departure_from_the_reference_is_detected(self) -> None:
        """A subsite dominated by a rare residue scores high against the reference."""

        rare_residue = int(np.argmin(SKEWED_BACKGROUND))
        counts = _deterministic_counts(rare_residue, 10_000)
        content = positional_information_content(counts, background=SKEWED_BACKGROUND)
        assert np.all(content > positional_information_content(counts))

    def test_counts_may_be_passed_as_the_reference(self) -> None:
        """An unnormalized reference is rescaled rather than rejected."""

        counts = _uniform_counts(10)
        scaled = positional_information_content(counts, background=SKEWED_BACKGROUND * 7.0)
        assert np.allclose(scaled, positional_information_content(counts, background=SKEWED_BACKGROUND))


class TestNullInformationContent:
    """Finite-sample floor of the divergence."""

    def test_null_is_positive_and_decays_once_sampling_dominates(self) -> None:
        """Beyond the prior-dominated regime the floor decays with depth."""

        null = expected_null_information_content(
            np.array([20, 50, 100, 1000]), n_samples=600, seed=1
        )
        assert np.all(null > 0.0)
        assert np.all(np.diff(null) < 0.0)

    def test_null_peaks_at_intermediate_depth(self) -> None:
        """The floor is not monotone: the smoothing prior flattens tiny samples.

        This is the regression guard for the documented non-monotonicity. The
        MEROPS serum panel has a median of 20 pooled cleavages per peptidase,
        which sits close to this worst-case peak.
        """

        depths = np.array([1, 3, 15, 100])
        null = expected_null_information_content(depths, n_samples=1500, seed=4)
        assert null[0] < null[1] < null[2]
        assert null[3] < null[2]

    def test_floor_is_computed_against_the_same_reference(self) -> None:
        """Counts drawn from the reference have no excess over the floor.

        The null draws from the reference and scores against it, so a subsite
        that genuinely follows the reference must sit at the floor regardless of
        how non-uniform that reference is.
        """

        generator = np.random.default_rng(11)
        draws = generator.multinomial(60, SKEWED_BACKGROUND, size=(1, N_SUBSITES)).astype(float)
        _, _, excess = information_content_excess(
            draws, n_samples=600, seed=12, background=SKEWED_BACKGROUND
        )
        assert np.all(np.abs(excess) < 0.35)

    def test_uniform_reference_adds_the_divergence_of_the_reference_itself(self) -> None:
        """Scoring proteome-like counts against a uniform reference adds KL(q||u).

        This is the regression guard for the reference choice. With enough
        observations the spurious excess converges to the divergence between the
        reference and the uniform distribution, which is a quantity known in
        closed form, so the test states the expected size rather than an
        arbitrary threshold.
        """

        generator = np.random.default_rng(11)
        draws = generator.multinomial(600, SKEWED_BACKGROUND, size=(1, N_SUBSITES)).astype(float)
        _, _, excess_uniform = information_content_excess(draws, n_samples=600, seed=12)
        _, _, excess_reference = information_content_excess(
            draws, n_samples=600, seed=12, background=SKEWED_BACKGROUND
        )
        divergence_of_the_reference = float(
            (SKEWED_BACKGROUND * np.log2(SKEWED_BACKGROUND * N_AMINO_ACIDS)).sum()
        )
        assert excess_uniform.mean() - excess_reference.mean() == pytest.approx(
            divergence_of_the_reference, abs=0.05
        )

    def test_zero_observations_give_zero(self) -> None:
        """A subsite with no observations has no null information content."""

        assert expected_null_information_content(np.array([0]), n_samples=50, seed=1)[0] == 0.0

    def test_uniform_counts_have_no_excess_over_the_null(self) -> None:
        """Counts drawn uniformly do not exceed the null by a meaningful margin."""

        generator = np.random.default_rng(7)
        draws = generator.multinomial(
            30, np.full(N_AMINO_ACIDS, 1.0 / N_AMINO_ACIDS), size=(1, N_SUBSITES)
        ).astype(float)
        observed, null, excess = information_content_excess(draws, n_samples=400, seed=2)
        assert observed.shape == null.shape == excess.shape == (1, N_SUBSITES)
        assert np.all(np.abs(excess) < 0.35)

    def test_specific_counts_exceed_the_null(self) -> None:
        """A genuinely specific subsite stands above its sampling floor."""

        _, _, excess = information_content_excess(_deterministic_counts(5, 30), n_samples=200, seed=3)
        assert np.all(excess > 2.0)


class TestBootstrap:
    """Resampling of specificity matrices."""

    def test_shape_normalization_and_determinism(self) -> None:
        """Replicates are normalized log-probabilities and reproducible by seed."""

        counts = _deterministic_counts(2, 20) + 1.0
        first = bootstrap_specificity_matrices(counts, n_replicates=8, seed=11)
        second = bootstrap_specificity_matrices(counts, n_replicates=8, seed=11)
        assert first.shape == (8, 1, N_SUBSITES, N_AMINO_ACIDS)
        assert np.allclose(np.exp(first).sum(axis=-1), 1.0)
        assert np.array_equal(first, second)

    def test_shallow_matrices_vary_more_than_deep_ones(self) -> None:
        """Bootstrap spread shrinks as the number of observed cleavages grows.

        The spread is compared in probability space. In log space every residue
        that was never observed keeps a large variance at any depth, because
        going from zero to one resampled count is a fixed multiplicative jump,
        so log-space spread does not separate the two regimes.
        """

        composition = np.zeros(N_AMINO_ACIDS)
        composition[[0, 1, 2, 3]] = [0.4, 0.3, 0.2, 0.1]
        shallow = np.tile(composition * 10, (1, N_SUBSITES, 1))
        deep = np.tile(composition * 2000, (1, N_SUBSITES, 1))
        shallow_spread = np.exp(bootstrap_specificity_matrices(shallow, n_replicates=48, seed=5)).std(axis=0)
        deep_spread = np.exp(bootstrap_specificity_matrices(deep, n_replicates=48, seed=5)).std(axis=0)
        assert shallow_spread.mean() > 5.0 * deep_spread.mean()

    def test_empty_subsite_is_passed_through(self) -> None:
        """A subsite with no observations is not invented by resampling."""

        counts = _deterministic_counts(1, 5)
        counts[:, 4, :] = 0.0
        replicates = bootstrap_specificity_matrices(counts, n_replicates=4, seed=9)
        uniform_log_probability = np.log(1.0 / N_AMINO_ACIDS)
        assert np.allclose(replicates[:, 0, 4, :], uniform_log_probability)


class TestProfileRedundancy:
    """Panel-level redundancy of specificity profiles."""

    def test_log_odds_shift_uniform_profile_to_zero(self) -> None:
        """A uniform profile carries no log-odds signal."""

        uniform = np.full((2, N_SUBSITES, N_AMINO_ACIDS), np.log(1.0 / N_AMINO_ACIDS))
        assert np.allclose(specificity_log_odds(uniform), 0.0)

    def test_identical_profiles_correlate_perfectly(self) -> None:
        """Duplicated matrices produce a correlation of one."""

        counts = np.concatenate([_deterministic_counts(2, 40) + 1.0] * 2, axis=0)
        log_probabilities = np.log(
            (counts + MEROPS_SMOOTHING_ALPHA)
            / (counts + MEROPS_SMOOTHING_ALPHA).sum(axis=-1, keepdims=True)
        )
        similarity = profile_similarity_matrix(log_probabilities)
        assert similarity.shape == (2, 2)
        assert similarity[0, 1] == pytest.approx(1.0)

    def test_clustering_merges_duplicates_and_separates_distinct_profiles(self) -> None:
        """Redundant proteases share a cluster while distinct ones do not."""

        counts = np.concatenate(
            [
                _deterministic_counts(2, 40) + 1.0,
                _deterministic_counts(2, 40) + 1.0,
                _deterministic_counts(15, 40) + 1.0,
            ],
            axis=0,
        )
        smoothed = counts + MEROPS_SMOOTHING_ALPHA
        log_probabilities = np.log(smoothed / smoothed.sum(axis=-1, keepdims=True))
        labels = cluster_profiles(profile_similarity_matrix(log_probabilities), threshold=0.8)
        assert labels[0] == labels[1]
        assert labels[2] != labels[0]

    def test_non_square_similarity_is_rejected(self) -> None:
        """A malformed similarity matrix raises instead of clustering silently."""

        with pytest.raises(ValueError):
            cluster_profiles(np.zeros((3, 4)))


class TestMeropsCodeLocation:
    """Locating the dataset that holds a peptidase matrix."""

    def test_index_maps_codes_to_datasets_with_counts(self) -> None:
        """Every located code carries a cleavage count per dataset."""

        locations = merops_code_locations()
        assert locations
        sample = next(iter(locations.values()))
        assert all(isinstance(key, int) and isinstance(value, int) for key, value in sample.items())

    def test_best_dataset_maximizes_the_cleavage_count(self) -> None:
        """The reported dataset is the one with the most evidence for that code."""

        locations = merops_code_locations()
        code = max(locations, key=lambda key: max(locations[key].values()))
        dataset_id, count = best_merops_dataset(code, locations=locations)
        assert count == max(locations[code].values())
        assert locations[code][dataset_id] == count

    def test_absent_code_reports_no_dataset(self) -> None:
        """A code without a matrix anywhere is reported as absent, not defaulted."""

        assert best_merops_dataset("Z99.999") == (None, 0)
