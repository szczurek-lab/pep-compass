"""Is the published architecture wrong, or only its training data?

The published half-life heads are small estimators on top of a frozen
representation backbone, fitted on 130 records. Refitting the same estimator
families on our own human-serum cohort separates two explanations for a weak
published result: an architecture that cannot represent the task, or a
training set that does not cover our measurement condition.

This is the only place in the module where anything is fitted. The backbone
stays frozen, so a refit costs one embedding pass plus seconds of estimator
fitting. Every number is produced under a sequence-similarity split, because a
random split over peptide analogues reports an optimistic figure.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model_benchmark.config import ARTIFACTS_DIR, SENSITIVITY_SEED

logger = logging.getLogger(__name__)

EMBEDDING_CACHE = ARTIFACTS_DIR / "embeddings"


def embed_cohort(cohort, backbone="esm", device=None, batch_size=16, use_cache=True):
    """Compute frozen pooled backbone embeddings for a cohort.

    :param pandas.DataFrame cohort: Aggregated cohort with a ``sequence`` column.
    :param str backbone: Backbone key, ``esm`` for the sequence representation.
    :param str device: Torch device string; CUDA when available by default.
    :param int batch_size: Peptides per forward pass.
    :param bool use_cache: Reuse a cached embedding matrix when the cohort matches.
    :return: Pooled embeddings of shape ``(N, D)``.
    :rtype: numpy.ndarray
    """
    import torch

    from model_benchmark.predictors.peptiverse import get_backbone, sequence_to_smiles

    sequences = cohort.sequence.tolist()
    ## REMARK: A content hash rather than the built-in hash, which is salted
    ## per process and would never produce a cache hit across runs.
    digest = hashlib.sha256("\n".join(sequences).encode("utf-8")).hexdigest()[:16]
    signature = f"{backbone}_{len(sequences)}_{digest}"
    cache_path = EMBEDDING_CACHE / f"{signature}.npy"
    if use_cache and cache_path.is_file():
        logger.info("Reusing cached embeddings %s", cache_path.name)
        return np.load(cache_path)

    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = get_backbone(backbone, resolved)
    items = (
        [sequence_to_smiles(sequence) for sequence in sequences]
        if backbone != "esm"
        else sequences
    )

    # Batched embedding of the frozen backbone
    blocks = []
    for start in range(0, len(items), batch_size):
        blocks.append(encoder.embed_pooled(items[start : start + batch_size]))  # (B, D)
    embeddings = np.vstack(blocks)  # (N, D)
    logger.info("Embedded %s peptides into %s dimensions", *embeddings.shape)

    if use_cache:
        EMBEDDING_CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
    return embeddings


def similarity_clusters(sequences, threshold=0.6):
    """Group sequences by pairwise similarity with greedy clustering.

    Two sequences join one cluster when their normalized Levenshtein
    similarity reaches the threshold, so analogues of one parent peptide cannot
    be split across training and validation folds.

    :param list[str] sequences: Peptide sequences.
    :param float threshold: Similarity above which sequences share a cluster.
    :return: Cluster label per sequence, shape ``(N,)``.
    :rtype: numpy.ndarray
    """
    from Levenshtein import ratio

    labels = np.full(len(sequences), -1, dtype=int)
    representatives = []
    ## REMARK: Longest first, so a cluster is anchored by its parent rather
    ## than by an arbitrary truncated analogue.
    order = sorted(range(len(sequences)), key=lambda index: -len(sequences[index]))
    for index in order:
        for cluster, representative in enumerate(representatives):
            if ratio(sequences[index], representative) >= threshold:
                labels[index] = cluster
                break
        else:
            labels[index] = len(representatives)
            representatives.append(sequences[index])
    logger.info(
        "Clustered %s peptides into %s groups at similarity %.2f",
        len(sequences),
        labels.max() + 1,
        threshold,
    )
    return labels


def _candidate_estimators(seed=SENSITIVITY_SEED):
    """Build the estimator families refitted on our cohort.

    The families mirror the published ones: a linear model, gradient boosting
    and a small dense head on the same frozen embeddings.

    :param int seed: Seed for the stochastic estimators.
    :return: Named estimator factories.
    :rtype: dict[str, callable]
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def ridge():
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))

    def boosting():
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=4,
        )

    def dense_head():
        ## REMARK: Early stopping is disabled on purpose: with about a hundred
        ## training peptides the validation split it carves out is too small to
        ## select a stopping point, which produced unstable fits. Regularization
        ## carries that role instead. A dense head on 1280 pooled dimensions
        ## still overfits this sample size and is reported for completeness.
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(256,),
                activation="relu",
                alpha=1.0,
                learning_rate_init=1e-3,
                max_iter=3000,
                early_stopping=False,
                random_state=seed,
            ),
        )

    return {"ridge": ridge, "xgboost": boosting, "dense_head": dense_head}


def cross_validate_refits(embeddings, targets, groups, folds=5, seed=SENSITIVITY_SEED):
    """Refit each estimator family under a sequence-similarity split.

    :param numpy.ndarray embeddings: Frozen embeddings, shape ``(N, D)``.
    :param numpy.ndarray targets: Measured ``log10`` seconds, shape ``(N,)``.
    :param numpy.ndarray groups: Similarity cluster labels, shape ``(N,)``.
    :param int folds: Number of grouped folds.
    :param int seed: Seed of the fold assignment.
    :return: Out-of-fold predictions and the per-family metric table.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    from sklearn.model_selection import GroupKFold

    from model_benchmark.metrics import baseline_metrics, regression_metrics

    features = np.asarray(embeddings, dtype=np.float32)  # (N, D)
    target = np.asarray(targets, dtype=float)  # (N,)
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))

    predictions = {name: np.full(target.shape, np.nan) for name in _candidate_estimators()}
    for fold, (train_index, test_index) in enumerate(
        splitter.split(features, target, groups=groups), start=1
    ):
        for name, factory in _candidate_estimators(seed=seed + fold).items():
            estimator = factory()
            estimator.fit(features[train_index], target[train_index])
            predictions[name][test_index] = estimator.predict(features[test_index])  # (n_test,)
        logger.info("Fold %s: %s train, %s test peptides", fold, len(train_index), len(test_index))

    # Out-of-fold scoring against the trivial baseline
    frame = pd.DataFrame({"measured_log10": target, "cluster": groups})
    records = [{"estimator": "median_baseline", **baseline_metrics(target)}]
    for name, values in predictions.items():
        frame[f"predicted_{name}"] = values
        records.append({"estimator": name, **regression_metrics(target, values)})
    return frame, pd.DataFrame(records)


def compare_with_published(refit_metrics, published_summary):
    """Place the refitted families next to the published artifact.

    :param pandas.DataFrame refit_metrics: Output of :func:`cross_validate_refits`.
    :param dict published_summary: Stored ``summary.json`` of a published artifact.
    :return: Comparison of error and rank correlation.
    :rtype: pandas.DataFrame
    """
    rows = refit_metrics.assign(source="refitted on our cohort, grouped split")
    published = pd.DataFrame(
        [
            {
                "estimator": published_summary["predictor"],
                "peptides": published_summary.get("peptides"),
                "mae_log10": published_summary.get("mae_log10"),
                "rmse_log10": published_summary.get("rmse_log10"),
                "r2": published_summary.get("r2"),
                "spearman_rho": published_summary.get("spearman_rho"),
                "median_error_factor": published_summary.get("median_error_factor"),
                "source": "published artifact, as distributed",
            }
        ]
    )
    columns = ["estimator", "source", "peptides", "mae_log10", "rmse_log10", "r2",
               "spearman_rho", "median_error_factor"]
    combined = pd.concat([published, rows], ignore_index=True)
    return combined[[name for name in columns if name in combined.columns]]


def refit_sensitivity(embeddings, targets, groups, sequences, probe_count=6, seed=SENSITIVITY_SEED):
    """Test whether a refitted head reacts to single substitutions.

    The refitted estimator is trained on all peptides and then probed with
    complete single-substitution scans, so its sensitivity can be compared with
    the published artifacts under the same test.

    :param numpy.ndarray embeddings: Frozen embeddings, shape ``(N, D)``.
    :param numpy.ndarray targets: Measured ``log10`` seconds, shape ``(N,)``.
    :param numpy.ndarray groups: Similarity cluster labels, shape ``(N,)``.
    :param list[str] sequences: Cohort sequences aligned with the embeddings.
    :param int probe_count: Peptides to scan.
    :param int seed: Seed of the estimator.
    :return: Per-peptide response of the refitted estimator.
    :rtype: pandas.DataFrame
    """
    import torch

    from model_benchmark.predictors.peptiverse import get_backbone

    estimator = _candidate_estimators(seed=seed)["ridge"]()
    estimator.fit(np.asarray(embeddings, dtype=np.float32), np.asarray(targets, dtype=float))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = get_backbone("esm", device)
    residues = tuple("ACDEFGHIKLMNPQRSTVWY")
    generator = np.random.default_rng(seed)
    probes = [sequences[index] for index in generator.choice(len(sequences), size=min(probe_count, len(sequences)), replace=False)]

    records = []
    for sequence in probes:
        variants = [
            sequence[:position] + replacement + sequence[position + 1 :]
            for position in range(len(sequence))
            for replacement in residues
            if replacement != sequence[position]
        ]
        items = [sequence] + variants
        features = np.vstack(
            [encoder.embed_pooled(items[start : start + 32]) for start in range(0, len(items), 32)]
        )  # (1 + V, D)
        predicted = estimator.predict(features)  # (1 + V,)
        delta = predicted[1:] - predicted[0]  # (V,)
        records.append(
            {
                "parent": sequence,
                "variants": len(variants),
                "parent_prediction_log10": float(predicted[0]),
                "median_absolute_delta_log10": float(np.median(np.abs(delta))),
                "max_absolute_delta_log10": float(np.abs(delta).max()),
            }
        )
    return pd.DataFrame(records)


def rank_agreement(frame, left, right):
    """Report the rank agreement between two prediction columns.

    :param pandas.DataFrame frame: Table holding both prediction columns.
    :param str left: First column name.
    :param str right: Second column name.
    :return: Spearman correlation between the two columns.
    :rtype: float
    """
    return float(spearmanr(frame[left], frame[right]).statistic)
