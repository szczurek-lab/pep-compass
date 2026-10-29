"""Numerical cost of scoring peptides with a published predictor.

Two quantities decide whether a predictor can serve as an optimisation oracle:
the latency of a single peptide, and the throughput of a batch. Both are
measured here on the machine that runs the study, including the share spent in
the representation backbone, because that share dominates.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import torch

from model_benchmark.config import (
    COST_BATCH_SIZES,
    COST_REPEATS,
    COST_SEQUENCE_LENGTH,
    COST_WARMUP_CALLS,
)

logger = logging.getLogger(__name__)

CANONICAL_RESIDUES = tuple("ACDEFGHIKLMNPQRSTVWY")


def synthetic_peptides(count, length=COST_SEQUENCE_LENGTH, seed=0):
    """Draw distinct random peptides for a timing run.

    Distinct sequences are required because adapters and backbones cache
    results per input, which would otherwise make a repeated batch free.

    :param int count: Number of peptides.
    :param int length: Residues per peptide.
    :param int seed: Seed of the random generator.
    :return: Peptide sequences.
    :rtype: list[str]
    """
    generator = np.random.default_rng(seed)
    residues = generator.choice(
        np.asarray(CANONICAL_RESIDUES), size=(count, length)
    )  # (count, length)
    return ["".join(row) for row in residues]


def _synchronize(device):
    """Wait for queued device work so timings are not optimistic.

    :param torch.device device: Device used by the predictor.
    :rtype: None
    """
    if getattr(device, "type", "cpu") == "cuda":
        torch.cuda.synchronize()


def measure_batch_cost(predictor, sizes=COST_BATCH_SIZES, repeats=COST_REPEATS, seed=0):
    """Time batched scoring at several batch sizes.

    :param model_benchmark.predictors.base.HalfLifePredictor predictor: Loaded adapter.
    :param tuple[int] sizes: Peptide counts per timed call.
    :param int repeats: Timed repetitions per size, each with fresh peptides.
    :param int seed: Base seed of the synthetic peptides.
    :return: One row per batch size with latency and throughput.
    :rtype: pandas.DataFrame
    """
    device = getattr(predictor, "device", torch.device("cpu"))

    ## Warm-up: first call pays lazy CUDA and kernel initialization
    for index in range(COST_WARMUP_CALLS):
        predictor.predict_log10_seconds(synthetic_peptides(2, seed=seed + 900 + index))
    _synchronize(device)

    records = []
    for size in sizes:
        ## Timed repetitions
        durations = []
        for repetition in range(repeats):
            peptides = synthetic_peptides(size, seed=seed + 1000 * (repetition + 1) + size)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            _synchronize(device)
            started = time.perf_counter()
            predictor.predict_log10_seconds(peptides)
            _synchronize(device)
            durations.append(time.perf_counter() - started)
        duration = np.asarray(durations, dtype=float)  # (repeats,)

        records.append(
            {
                "predictor": predictor.name,
                "device": str(device),
                "batch_size": size,
                "seconds_median": float(np.median(duration)),
                "seconds_min": float(duration.min()),
                "seconds_per_peptide": float(np.median(duration) / size),
                "peptides_per_second": float(size / np.median(duration)),
                "peak_gpu_megabytes": float(torch.cuda.max_memory_allocated() / 1e6)
                if device.type == "cuda"
                else float("nan"),
            }
        )
        logger.info(
            "%s batch %s: %.3f s (%.1f peptides/s)",
            predictor.name,
            size,
            records[-1]["seconds_median"],
            records[-1]["peptides_per_second"],
        )
    return pd.DataFrame(records)


def measure_unbatched_cost(predictor, count=100, seed=0):
    """Time the per-peptide calling pattern of the published inference script.

    :param model_benchmark.predictors.base.HalfLifePredictor predictor: Loaded adapter.
    :param int count: Peptides scored one by one.
    :param int seed: Seed of the synthetic peptides.
    :return: Latency statistics of single-peptide calls.
    :rtype: dict
    """
    device = getattr(predictor, "device", torch.device("cpu"))
    peptides = synthetic_peptides(count, seed=seed + 7)
    predictor.predict_log10_seconds(peptides[:1])
    _synchronize(device)

    durations = []
    for peptide in peptides:
        started = time.perf_counter()
        predictor.predict_log10_seconds([peptide])
        _synchronize(device)
        durations.append(time.perf_counter() - started)
    duration = np.asarray(durations, dtype=float)  # (count,)

    return {
        "predictor": predictor.name,
        "device": str(device),
        "peptides": count,
        "seconds_per_peptide_median": float(np.median(duration)),
        "seconds_per_peptide_mean": float(duration.mean()),
        "peptides_per_second": float(1.0 / np.median(duration)),
        "total_seconds": float(duration.sum()),
    }


def projected_cost(batch_table, unbatched, targets=(1, 1000, 100_000)):
    """Project wall-clock cost for peptide counts of practical interest.

    The projection uses the largest measured batch size for the batched path,
    which is the regime an optimisation loop would run in.

    :param pandas.DataFrame batch_table: Output of :func:`measure_batch_cost`.
    :param dict unbatched: Output of :func:`measure_unbatched_cost`.
    :param tuple[int] targets: Peptide counts to project.
    :return: Projected seconds per path and the speed-up from batching.
    :rtype: pandas.DataFrame
    """
    largest = batch_table.sort_values("batch_size").iloc[-1]
    records = []
    for count in targets:
        batched_seconds = count * largest.seconds_per_peptide
        single_seconds = count * unbatched["seconds_per_peptide_median"]
        records.append(
            {
                "predictor": largest.predictor,
                "peptides": count,
                "batched_seconds": float(batched_seconds),
                "unbatched_seconds": float(single_seconds),
                "batching_speedup": float(single_seconds / batched_seconds)
                if batched_seconds > 0
                else float("nan"),
                "reference_batch_size": int(largest.batch_size),
            }
        )
    return pd.DataFrame(records)
