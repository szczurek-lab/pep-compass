"""Does a predictor react to a single amino-acid substitution?

A predictor is only usable inside peptide optimisation if its output changes
when one residue changes, if the change depends on which position was edited,
and if the direction of the change matches measurements. The three questions
are tested separately:

* a complete single-substitution scan, which measures whether and how much the
  output moves at all;
* a permutation test, which keeps the composition and only reorders residues,
  separating a positional model from a composition-only model;
* measured parent-variant pairs at Hamming distance one, which test the sign
  and the size of the predicted change against experiment.

For differentiable adapters the finite-difference scan is additionally
compared with input gradients.
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model_benchmark.config import PERMUTATION_REPEATS, SENSITIVITY_SEED

logger = logging.getLogger(__name__)

CANONICAL_RESIDUES = tuple("ACDEFGHIKLMNPQRSTVWY")


def mutational_scan(predictor, sequence):
    """Score every single substitution of one peptide.

    :param model_benchmark.predictors.base.HalfLifePredictor predictor: Loaded adapter.
    :param str sequence: Parent peptide sequence.
    :return: One row per variant with its signed change in ``log10`` seconds.
    :rtype: pandas.DataFrame
    :raises ValueError: If the sequence contains a non-canonical residue.
    """
    if not set(sequence) <= set(CANONICAL_RESIDUES):
        raise ValueError(f"Sequence {sequence!r} leaves the canonical alphabet")

    # Variant enumeration
    variants, positions, originals, replacements = [], [], [], []
    for position, residue in enumerate(sequence):
        for replacement in CANONICAL_RESIDUES:
            if replacement == residue:
                continue
            variants.append(sequence[:position] + replacement + sequence[position + 1 :])
            positions.append(position)
            originals.append(residue)
            replacements.append(replacement)

    # Batched scoring of parent and variants
    predictions = predictor.predict_log10_seconds([sequence] + variants)  # (1 + V,)
    parent = float(predictions[0])
    frame = pd.DataFrame(
        {
            "predictor": predictor.name,
            "parent": sequence,
            "position": positions,
            "wild_type_residue": originals,
            "variant_residue": replacements,
            "variant": variants,
            "prediction_log10": predictions[1:],
        }
    )
    frame["parent_prediction_log10"] = parent
    frame["delta_log10"] = frame.prediction_log10 - parent
    return frame


def scan_summary(scan, noise_floor=0.0):
    """Summarize one single-substitution scan.

    :param pandas.DataFrame scan: Output of :func:`mutational_scan`.
    :param float noise_floor: Change in ``log10`` seconds below which a
        difference is indistinguishable from measurement noise.
    :return: Response size, positional structure and degeneracy flags.
    :rtype: dict
    """
    delta = scan.delta_log10.to_numpy(dtype=float)  # (V,)
    absolute = np.abs(delta)  # (V,)

    ## Positional structure
    ## REMARK: A model that only reads composition still moves when a residue
    ## is replaced, but the size of that move barely depends on the position.
    ## The ratio of between-position to total variance separates the two.
    by_position = scan.groupby("position").delta_log10
    position_means = by_position.mean().to_numpy(dtype=float)  # (L,)
    between = float(np.var(position_means)) if position_means.size > 1 else 0.0
    total = float(np.var(delta)) if delta.size > 1 else 0.0

    return {
        "predictor": scan.predictor.iloc[0],
        "parent": scan.parent.iloc[0],
        "length": int(scan.position.nunique()),
        "variants": int(len(scan)),
        "parent_prediction_log10": float(scan.parent_prediction_log10.iloc[0]),
        "max_absolute_delta_log10": float(absolute.max()),
        "median_absolute_delta_log10": float(np.median(absolute)),
        "std_delta_log10": float(delta.std(ddof=0)),
        "max_fold_change": float(10.0 ** absolute.max()),
        "identical_prediction_share": float((absolute == 0.0).mean()),
        "below_noise_floor_share": float((absolute < noise_floor).mean())
        if noise_floor > 0
        else float("nan"),
        "position_variance_share": float(between / total) if total > 0 else float("nan"),
        "most_sensitive_position": int(by_position.apply(lambda values: values.abs().max()).idxmax()),
        "stabilizing_variants": int((delta > 0).sum()),
        "destabilizing_variants": int((delta < 0).sum()),
    }


def permutation_sensitivity(predictor, sequence, repeats=PERMUTATION_REPEATS, seed=SENSITIVITY_SEED):
    """Compare the response to reordering with the response to substitution.

    A permutation preserves amino-acid composition exactly. A predictor whose
    representation ignores order returns the parent prediction for every
    permutation.

    :param model_benchmark.predictors.base.HalfLifePredictor predictor: Loaded adapter.
    :param str sequence: Parent peptide sequence.
    :param int repeats: Number of distinct permutations.
    :param int seed: Seed of the random generator.
    :return: Spread of predictions over permutations of one composition.
    :rtype: dict
    """
    generator = np.random.default_rng(seed)
    residues = np.asarray(list(sequence))

    ## Distinct permutations of the same composition
    permutations, attempts = [], 0
    while len(permutations) < repeats and attempts < 50 * repeats:
        attempts += 1
        candidate = "".join(generator.permutation(residues))
        if candidate != sequence and candidate not in permutations:
            permutations.append(candidate)
    if not permutations:
        return {"predictor": predictor.name, "parent": sequence, "permutations": 0}

    predictions = predictor.predict_log10_seconds([sequence] + permutations)  # (1 + P,)
    parent = float(predictions[0])
    delta = predictions[1:] - parent  # (P,)
    return {
        "predictor": predictor.name,
        "parent": sequence,
        "permutations": len(permutations),
        "max_absolute_permutation_delta_log10": float(np.abs(delta).max()),
        "median_absolute_permutation_delta_log10": float(np.median(np.abs(delta))),
        "std_permutation_delta_log10": float(delta.std(ddof=0)),
    }


def gradient_agreement(predictor, sequence, scan):
    """Compare input gradients with the finite-difference scan per position.

    :param model_benchmark.predictors.base.HalfLifePredictor predictor: Loaded adapter.
    :param str sequence: Parent peptide sequence.
    :param pandas.DataFrame scan: Output of :func:`mutational_scan`.
    :return: Gradient norms, their positional agreement, or an unavailability note.
    :rtype: dict
    """
    try:
        gradient = predictor.per_residue_gradient(sequence)  # (L,)
    except NotImplementedError as error:
        return {"predictor": predictor.name, "available": False, "reason": str(error)}

    positional = (
        scan.groupby("position").delta_log10.apply(lambda values: values.abs().max()).to_numpy()
    )  # (L,)
    if gradient.shape != positional.shape:
        return {
            "predictor": predictor.name,
            "available": False,
            "reason": f"gradient length {gradient.shape[0]} differs from scan length {positional.shape[0]}",
        }

    return {
        "predictor": predictor.name,
        "available": True,
        "parent": sequence,
        "gradient_norm_sum": float(gradient.sum()),
        "gradient_norm_max": float(gradient.max()),
        "nonzero_gradient_share": float((gradient > 0).mean()),
        "spearman_gradient_vs_scan": float(spearmanr(gradient, positional).statistic),
    }


def empirical_substitution_pairs(records, max_length=50):
    """Find measured parent-variant pairs differing in exactly one residue.

    Pairs are formed only inside one assay condition, because half-lives
    measured in different species or matrices are not comparable.

    :param pandas.DataFrame records: Non-aggregated cohort records.
    :param int max_length: Longest sequence considered.
    :return: Measured pairs with their condition and measured change.
    :rtype: pandas.DataFrame
    """
    frame = records.loc[records.sequence.str.len() <= max_length].copy()
    frame["condition"] = (
        frame.species.fillna("?") + "|" + frame.matrix.fillna("?") + "|" + frame.in_vivo_in_vitro.fillna("?")
    )

    # Pair enumeration within each condition and sequence length
    pairs = []
    for condition, group in frame.groupby("condition"):
        measured = group.groupby("sequence").log10_half_life.mean()
        by_length = defaultdict(list)
        for sequence in measured.index:
            by_length[len(sequence)].append(sequence)
        for sequences in by_length.values():
            for left, right in itertools.combinations(sorted(sequences), 2):
                differences = [
                    (index, first, second)
                    for index, (first, second) in enumerate(zip(left, right))
                    if first != second
                ]
                if len(differences) != 1:
                    continue
                position, parent_residue, variant_residue = differences[0]
                pairs.append(
                    {
                        "condition": condition,
                        "parent": left,
                        "variant": right,
                        "position": position,
                        "parent_residue": parent_residue,
                        "variant_residue": variant_residue,
                        "parent_measured_log10": float(measured[left]),
                        "variant_measured_log10": float(measured[right]),
                        "measured_delta_log10": float(measured[right] - measured[left]),
                    }
                )
    result = pd.DataFrame(pairs)
    logger.info("Found %s measured single-substitution pairs", len(result))
    return result


def score_empirical_pairs(predictor, pairs, minimum_effect=0.0):
    """Test predicted changes against measured changes on parent-variant pairs.

    :param model_benchmark.predictors.base.HalfLifePredictor predictor: Loaded adapter.
    :param pandas.DataFrame pairs: Output of :func:`empirical_substitution_pairs`.
    :param float minimum_effect: Restrict scoring to pairs whose measured change
        exceeds this size in ``log10`` seconds, excluding pairs where the
        measurement itself cannot resolve a direction.
    :return: Per-pair predictions and the aggregate agreement record.
    :rtype: tuple[pandas.DataFrame, dict]
    """
    if pairs.empty:
        return pairs, {"predictor": predictor.name, "pairs": 0}

    selected = pairs.loc[pairs.measured_delta_log10.abs() >= minimum_effect].copy()
    if selected.empty:
        return selected, {"predictor": predictor.name, "pairs": 0, "minimum_effect": minimum_effect}

    # Batched scoring of both sides
    unique = sorted(set(selected.parent) | set(selected.variant))
    predicted = dict(zip(unique, predictor.predict_log10_seconds(unique)))
    selected["parent_predicted_log10"] = selected.parent.map(predicted)
    selected["variant_predicted_log10"] = selected.variant.map(predicted)
    selected["predicted_delta_log10"] = (
        selected.variant_predicted_log10 - selected.parent_predicted_log10
    )

    measured = selected.measured_delta_log10.to_numpy(dtype=float)  # (P,)
    estimated = selected.predicted_delta_log10.to_numpy(dtype=float)  # (P,)
    agreement = np.sign(measured) == np.sign(estimated)  # (P,)

    record = {
        "predictor": predictor.name,
        "pairs": int(len(selected)),
        "minimum_effect": minimum_effect,
        "sign_agreement": float(agreement.mean()),
        "spearman_delta": float(spearmanr(measured, estimated).statistic),
        "mae_delta_log10": float(np.abs(estimated - measured).mean()),
        "median_absolute_measured_delta": float(np.median(np.abs(measured))),
        "median_absolute_predicted_delta": float(np.median(np.abs(estimated))),
        ## REMARK: A model can agree on direction and still be useless for
        ## optimisation if its predicted effects are far smaller than the
        ## measured ones; the ratio makes that visible.
        "predicted_to_measured_effect_ratio": float(
            np.median(np.abs(estimated)) / np.median(np.abs(measured))
        )
        if np.median(np.abs(measured)) > 0
        else float("nan"),
    }
    return selected, record
