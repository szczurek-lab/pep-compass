from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from poli.core.black_box_information import BlackBoxInformation
from poli.core.exceptions import ObserverNotInitializedError
from poli.core.util.abstract_observer import AbstractObserver

from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)


class ListObserver(AbstractObserver):
    """An observer that stores every observed result in memory."""

    def __init__(self, maximize: bool):
        self.has_been_initialized = False
        super().__init__()
        self.maximize = maximize

    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: dict,
        seed: int,
        encoder_decoder: HydrAMPEncoderDecoder | None = None,
    ) -> object:
        """Initialize the observer and clear its in-memory result lists."""
        self.info = problem_setup_info
        self.seed = seed
        self.encoder_decoder = encoder_decoder

        self.times: list[datetime] = []
        self.sequences: list[str | None] = []
        self.scores: list[float] = []
        self.latent_points: list[np.ndarray | None] = []

        self.best_score = float("-inf") if self.maximize else float("inf")
        self.best_sequence = None
        self.has_been_initialized = True

    def _validate_input(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError(f"x should be 2D, got {x.ndim}D instead.")
        if y.ndim != 2:
            raise ValueError(f"y should be 2D, got {y.ndim}D instead.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y should have the same number of samples, got {x.shape[0]} and {y.shape[0]} respectively."
            )

    def _ensure_proper_shape(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x.reshape(-1, 1)
        return x

    def observe(self, x: np.ndarray, y: np.ndarray, context: dict = None) -> None:
        if not self.has_been_initialized:
            raise ObserverNotInitializedError(
                "The observer has not been initialized. Please call `initialize_observer` first."
            )
        x = self._ensure_proper_shape(x)
        self._validate_input(x, y)

        if isinstance(x[0, 0], str):
            sequences = ["".join(x_i) for x_i in x]
            latent_points = [None for _ in range(len(sequences))]
        elif self.encoder_decoder is not None:
            sequences = self.encoder_decoder.decode_peptides(
                torch.tensor(x).to(self.encoder_decoder.device).to(torch.float32)
            )
            latent_points = x
        else:
            latent_points = x
            sequences = [None for _ in range(len(latent_points))]

        scores = [y_i for y_i in y.flatten()]

        if self.maximize:
            self.best_score = max(self.best_score, max(scores))
        else:
            self.best_score = min(self.best_score, min(scores))

        if self.best_score in scores:
            self.best_sequence = sequences[scores.index(self.best_score)]

        self.append_results(sequences, scores, latent_points)

        print(
            f"Observer: Best score so far: {self.best_score} for sequence {self.best_sequence}"
        )

    def append_results(
        self,
        sequences: list[str | None],
        scores: list[float],
        latent_points: list[np.ndarray | None],
    ) -> None:
        """Append one entry per observed sample to the in-memory lists."""
        self.times.extend(datetime.now() for _ in sequences)
        self.sequences.extend(sequences)
        self.scores.extend(scores)
        self.latent_points.extend(latent_points)
