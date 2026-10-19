"""Scoped deterministic random state for backend execution."""

from __future__ import annotations

import random
from contextlib import contextmanager, nullcontext
from threading import RLock
from typing import TYPE_CHECKING

import numpy as np
import torch

from pep_compass.utils.logger import get_custom_logger
from pep_compass.utils.validation import validate_integer

if TYPE_CHECKING:
    from collections.abc import Iterator


logger = get_custom_logger(__name__)

_run_lock = RLock()


def seed_generators(seed: int) -> None:
    """Seed Python, legacy NumPy and Torch generators consistently.

    :param seed: Non-negative integer seed.
    :raises ValueError: If the seed is invalid.

    :type seed: int
    :rtype: None
    """
    validate_integer(seed, "seed")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**63))


@contextmanager
def random_scope(
    seed: int | None, *, serialize: bool = True, preserve_unseeded: bool = False
) -> Iterator[None]:
    """Seed and restore process RNGs and deterministic Torch kernel settings.

    :param seed: Seed for a reproducible execution, or null to retain the stream.
    :param serialize: Serialize top-level runs sharing process-global generators.
    :param preserve_unseeded: Snapshot state without reseeding, for isolated replay.
    :yields: Control while the requested random state is active.

    REMARK: Branch execution is ordered by Parallel because model adapters and
    third-party libraries may consume process-global RNGs and mutable state.
    Separate runtime subprocesses provide independent concurrent runs.

    :type seed: int | None
    :type serialize: bool
    :type preserve_unseeded: bool
    :return: Result of this operation.
    :rtype: Iterator[None]
    """
    with _run_lock if serialize else nullcontext():
        if seed is None and not preserve_unseeded:
            yield
            return
        logger.debug("Entering deterministic RNG scope seed=%s.", seed)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        benchmark = torch.backends.cudnn.benchmark
        cudnn_deterministic = torch.backends.cudnn.deterministic
        devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=devices):
            try:
                if seed is not None:
                    seed_generators(seed)
                    torch.use_deterministic_algorithms(True)
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                yield
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
                torch.backends.cudnn.benchmark = benchmark
                torch.backends.cudnn.deterministic = cudnn_deterministic
