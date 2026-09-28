# Create new environment with
# pip install git+https://github.com/MachineLearningLifeScience/poli.git@dev
# pip install 'poli-baselines[lambo2] @ git+https://github.com/MachineLearningLifeScience/poli-baselines.git'
# pip install "seqme[aa_descriptors]"
# pip install pytorch-cortex
# pip install "tensorflow[and-cuda]==2.20"

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional
from uuid import uuid4

import numpy as np
from poli.core.black_box_information import BlackBoxInformation
from poli.core.exceptions import ObserverNotInitializedError
from poli.core.util.abstract_observer import AbstractObserver
from poli_baselines.core.abstract_solver import AbstractBlackBox
from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2

from battleamp.BattleAMPPredictor import PredictorBattleAMP

OUTPUT_PATH = "./results/lambo"
DEVICE = "cuda:1"

class BattleAMPBlackBox(AbstractBlackBox):
    def __init__(
        self,
        *,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        
        self.device = device
        self.battleamp_predictor = PredictorBattleAMP(device=device)
        
        # BattleAMP returns a single prediction value, so no aggregation needed
        self.peptide_scorer = lambda x: -np.log2(self.battleamp_predictor.predict(x).flatten())

        self.maximize = False

        self.cache = []

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="BattleAMP",
            max_sequence_length=25,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=list("ACDEFGHIKLMNPQRSTVWY"),
            log_transform_recommended=False,
            discrete=True,
            padding_token=" ",
        )

    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        sequences = ["".join(seq) for seq in x]
        predictions = self.peptide_scorer(sequences)
        
        return predictions.reshape(-1, 1)

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
        else:
            latent_points = x
            sequences = [None for _ in range(len(latent_points))]

        scores = [-y_i for y_i in y.flatten()]

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


black_box = BattleAMPBlackBox(device=DEVICE)

observer = CSVObserver(maximize=False)
black_box.set_observer(observer)

proteins = {
    "middle-1": ("FLYKWWIRIGRLKL", 5),
    "jurand-4": ("KYCRRFRWLTFRWL", 5),
    "jurand-2": ("KFRNRHRWKFKLIFRN", 5),
    "jurand-7": ("KKYWLIRKWIRLWFLT", 5),
    "mammuthusin-3": ("KTLKIIRLLF", 5),
    "hydrodamin-2": ("RMARNLVRYVQGLKKKKVI", 5),
}

for name, (sequence, num) in proteins.items():
    for i in range(num):
        rng_seed = int(time())  # Create a unique rng_seed for each iteration
        observer.initialize_observer(
            black_box.get_black_box_info(),
            {"experiment_id": f"{sequence}_{rng_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "experiment_path": OUTPUT_PATH},
            rng_seed,
        )

        x0 = np.array([list(sequence)], dtype='<U1')

        solver = LaMBO2(
            black_box=black_box,
            x0=x0,
        )

        result = solver.solve(max_iter=10)

        all_y = np.concatenate(solver.history_for_training["y"], axis=0)
        all_x = np.concatenate(solver.history_for_training["x"], axis=0)
