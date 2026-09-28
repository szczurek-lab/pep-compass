from dataclasses import dataclass
import json
from pathlib import Path
from time import time
from uuid import uuid4

import numpy as np

from poli.core.black_box_information import BlackBoxInformation
from poli.core.exceptions import ObserverNotInitializedError
from poli.core.util.abstract_observer import AbstractObserver
from datetime import datetime

import torch

from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
    HydrAMPEncoderDecoder,
)


@dataclass
class CSVObserverInitInfo:
    """Initialization information for the LoggingObserver."""

    experiment_id: str
    results_path: str | Path


class CSVObserver(AbstractObserver):
    """
    A simple observer that logs to a CSV file, appending rows on each query.
    """

    def __init__(self,  maximize: bool):
        self.has_been_initialized = False
        super().__init__()
        self.maximize = maximize

    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: CSVObserverInitInfo | dict,
        seed: int,
        encoder_decoder: HydrAMPEncoderDecoder | None = None,
    ) -> object:
        """
        Initializes the observer with the given information.

        Parameters
        ----------
        black_box_info : BlackBoxInformation
            The information about the black box.
        caller_info : dict | CSVObserverInitInfo
            Information used for logging. If a dictionary, it should contain the
            keys `experiment_id` and `experiment_path`.
        seed : int
            The seed used for the experiment. This is only logged, not used.
        """
        self.info = problem_setup_info
        self.seed = seed
        self.encoder_decoder = encoder_decoder
        self.unique_id = f"{uuid4()}"[:8]

        if isinstance(caller_info, CSVObserverInitInfo):
            caller_info = caller_info.__dict__

        self.all_results_path = Path(
            caller_info.get("experiment_path", "./poli_results")
        )
        self.experiment_path = self.all_results_path / problem_setup_info.name
        self.experiment_path.mkdir(exist_ok=True, parents=True)

        self.experiment_id = caller_info.get(
            "experiment_id",
            f"{int(time())}_experiment_{problem_setup_info.name}_{seed}_{self.unique_id}",
        )

        self.csv_file_path = self.experiment_path / f"{self.experiment_id}.csv"
        self.save_header()

        self.best_score = float("-inf") if self.maximize else float("inf")
        self.best_sequence = None

        self.has_been_initialized = True

    def _make_folder_for_experiment(self):
        self.experiment_path.mkdir(exist_ok=True, parents=True)

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

    def save_header(self):
        self._make_folder_for_experiment()
        with open(self.csv_file_path, "w") as f:
            f.write("time,sequence,score,latent_point\n")

    def append_results(
        self, x: list[str], y: list[float], latent_points: list[list[float]]
    ):
        with open(self.csv_file_path, "a") as f:
            for x_i, y_i, latent_point in zip(x, y, latent_points):
                formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                latent_str = (
                    json.dumps(latent_point.tolist())
                    if latent_point is not None
                    else ""
                )
                f.write(f"{formatted_time},{x_i},{y_i},{latent_str}\n")
                print(f"{formatted_time},{x_i},{y_i},{latent_str}")
