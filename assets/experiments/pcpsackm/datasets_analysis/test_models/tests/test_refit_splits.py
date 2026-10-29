"""Tests of the sequence-similarity split used by the refit experiment."""

from __future__ import annotations

import numpy as np

from model_benchmark.refit import similarity_clusters


def test_identical_sequences_share_a_cluster():
    """Duplicate sequences never land in different folds."""
    sequences = ["GLFDIIKKIAESF", "GLFDIIKKIAESF", "KLAKLAKKLAKLAK"]
    labels = similarity_clusters(sequences, threshold=0.6)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_close_analogues_share_a_cluster():
    """A single substitution keeps an analogue in its parent's cluster."""
    sequences = ["GLFDIIKKIAESF", "GLFDIIKKIAESW", "GLFDIIKKIAERF"]
    labels = similarity_clusters(sequences, threshold=0.6)
    assert len(set(labels.tolist())) == 1


def test_unrelated_sequences_are_separated():
    """Sequences below the threshold receive distinct clusters."""
    sequences = ["GLFDIIKKIAESF", "WWWWWWWW", "PPPPPPPPPP"]
    labels = similarity_clusters(sequences, threshold=0.6)
    assert len(set(labels.tolist())) == 3


def test_every_sequence_is_assigned():
    """Clustering leaves no sequence unlabelled."""
    sequences = ["GLFDIIK", "GLFDIIR", "WWKKGG", "AAGGKK", "YGGFL", "YGGFM"]
    labels = similarity_clusters(sequences, threshold=0.6)
    assert labels.shape == (len(sequences),)
    assert (labels >= 0).all()
    assert set(np.unique(labels).tolist()) == set(range(labels.max() + 1))


def test_threshold_controls_granularity():
    """A stricter threshold cannot produce fewer clusters than a looser one."""
    sequences = ["GLFDIIKKIAESF", "GLFDIIKKIAESW", "GLFDIIKKAAESW", "KLAKLAKKLAKLAK"]
    loose = similarity_clusters(sequences, threshold=0.5).max() + 1
    strict = similarity_clusters(sequences, threshold=0.95).max() + 1
    assert strict >= loose
