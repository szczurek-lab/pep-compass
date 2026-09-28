# Create new environment with
# pip install git+https://github.com/MachineLearningLifeScience/poli.git@dev
# pip install 'poli-baselines[lambo2] @ git+https://github.com/MachineLearningLifeScience/poli-baselines.git'
# pip install "seqme[aa_descriptors]"
# pip install pytorch-cortex

from __future__ import annotations

import csv
import glob
import json
import math
import os
import os.path
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time
from typing import cast
from uuid import uuid4

import numpy as np
import torch
from numpy.typing import NDArray
from poli.core.abstract_black_box import AbstractBlackBox, BlackBoxInformation
from poli.core.black_box_information import BlackBoxInformation
from poli.core.exceptions import ObserverNotInitializedError
from poli.core.util.abstract_observer import AbstractObserver
from poli_baselines.core.abstract_solver import AbstractBlackBox
from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2
from tqdm import tqdm

OUTPUT_PATH = "./results/lambo"
DEVICE = "cuda:1"


def make_vocab():
    # 0: pad
    # 1: start
    # 2: end

    word2idx = {}
    idx2word = {}

    word2idx["0"] = 0
    word2idx["1"] = 1
    word2idx["2"] = 2

    word2idx["A"] = 3
    word2idx["C"] = 4
    word2idx["D"] = 5
    word2idx["E"] = 6
    word2idx["F"] = 7
    word2idx["G"] = 8
    word2idx["H"] = 9
    word2idx["I"] = 10
    word2idx["K"] = 11
    word2idx["L"] = 12
    word2idx["M"] = 13
    word2idx["N"] = 14
    word2idx["P"] = 15
    word2idx["Q"] = 16
    word2idx["R"] = 17
    word2idx["S"] = 18
    word2idx["T"] = 19
    word2idx["V"] = 20
    word2idx["W"] = 21
    word2idx["Y"] = 22

    for key, value in word2idx.items():
        idx2word[value] = key

    return word2idx, idx2word


def AAindex(path, word2idx):
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        AAindex_dict = {}
        AAindex_matrix = []
        skip = 1
        for row in reader:
            if skip == 1:
                skip = 0
                header = np.array(row)[1:].tolist()
                continue
            tmp = []
            for j in np.array(row)[1:]:
                try:
                    tmp.append(float(j))
                except:
                    tmp.append(0)
            AAindex_matrix.append(np.array(tmp))

        dim = np.shape(AAindex_matrix)[0]
        AAindex_matrix = np.array(AAindex_matrix)
        for i in range(len(header)):
            AAindex_dict[header[i]] = AAindex_matrix[:, i]

    # print (AAindex_matrix)
    emb = np.zeros((len(word2idx), dim))
    for key, value in word2idx.items():
        if key in AAindex_dict:
            emb[value] = AAindex_dict[key]
        else:
            pass
    return emb, AAindex_dict


def onehot_encoding(seq_list_, max_len, word2idx):
    # 0: pad
    # 1: start
    # 2: end
    seq_list = [i for i in seq_list_]
    X = np.zeros((len(seq_list), max_len)).astype(int)

    AA_mask = []
    nonAA_mask = []

    for i in range(len(seq_list)):
        if len(seq_list[i]) >= max_len - 2:
            a_seq = "1" + seq_list[i][: max_len - 2].upper() + "2"
        else:
            a_seq = "1" + seq_list[i].upper() + "2"

        if len(a_seq) > max_len:
            iter_num = max_len
        else:
            iter_num = len(a_seq)

        for j in range(iter_num):
            if a_seq[j] not in word2idx:
                continue
            else:
                X[i, j] = word2idx[a_seq[j]]

        tmp = np.zeros(max_len)
        tmp[1 : iter_num + 1] = 1
        AA_mask.append(tmp.astype(int))
        nonAA_mask.append((1 - tmp).astype(int))

    return np.array(X)  # , np.array(AA_mask), np.array(nonAA_mask)

class PredictorAPEX:
    def __init__(self, device="cpu", batch_size=3000, path="default"):
        self.device = device
        self.path = path
        if path == "default":
            self.pathogen_list = [
                "A. baumannii ATCC 19606",
                "E. coli ATCC 11775",
                "E. coli AIG221",
                "E. coli AIG222",
                "K. pneumoniae ATCC 13883",
                "P. aeruginosa PA01",
                "P. aeruginosa PA14",
                "S. aureus ATCC 12600",
                "S. aureus (ATCC BAA-1556) - MRSA",
                "vancomycin-resistant E. faecalis ATCC 700802",
                "vancomycin-resistant E. faecium ATCC 700221",
            ]
        elif path == "all":
            self.pathogen_list = [
                "A. baumannii ATCC 19606",
                "E. coli ATCC 11775",
                "E. coli AIG221",
                "E. coli AIG222",
                "K. pneumoniae ATCC 13883",
                "P. aeruginosa PA01",
                "P. aeruginosa PA14",
                "S. aureus ATCC 12600",
                "S. aureus (ATCC BAA-1556) - MRSA",
                "vancomycin-resistant E. faecalis ATCC 700802",
                "vancomycin-resistant E. faecium ATCC 700221",
                # Additional bacteria from APEX_FULL
                "A. muciniphila ATCC BAA-835",
                "B. fragilis ATCC25285",
                "B. vulgatus ATCC8482",
                "C. aerofaciens ATCC25986",
                "C. scindens ATCC35704",
                "B. thetaiotaomicron ATCC29148",
                "B. thetaiotaomicron Complemmented",
                "B. thetaiotaomicron Mutant",
                "B. uniformis ATCC8492",
                "B. eggerthi ATCC27754",
                "C. spiroforme ATCC29900",
                "P. distasonis ATCC8503",
                "P. copri DSMZ18205",
                "B. ovatus ATCC8483",
                "E. rectale ATCC33656",
                "C. symbiosum",
                "R. obeum",
                "R. torques",
                "E. coli Nissle",
                "Salmonella enterica ATCC 9150 (BEIRES NR-515)",
                "Salmonella enterica (BEIRES NR-170)",
                "Salmonella enterica ATCC 9150 (BEIRES NR-174)",
                "L. monocytogenes ATCC 19111 (BEIRES NR-106)",
            ]

        else:
            raise ValueError("Path option not recognized. Use 'default' or 'all'.")

        self.max_len = 52  # maximum seq length; 52 = start character + maximum peptide length (50 aa) + end character; longer peptides will be truncated
        self.word2idx, self.idx2word = make_vocab()  # make amino acid vocabulary
        # emb, AAindex_dict = AAindex('./aaindex1.csv', word2idx) #make amino acid embeddings

        # Load pretrained APEX models (8 in total)
        # Use custom pickle module to handle old module name references
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.APEX_models = []
        if path == "default":
            if not Path(f"{self.file_dir}/../src/pep_compass/models/apex/APEX_pathogen_models").exists():
                raise FileNotFoundError(
                    f"Directory {self.file_dir}/../src/pep_compass/models/apex/APEX_pathogen_models with APEX pathogen models not found. "
                    "Please copy the folder from https://github.com/Yimeng-Zeng/APEXGo/tree/main/optimization/apex_oracle/APEX_pathogen_models"
                )

            for a_model in glob.glob(f"{self.file_dir}/../src/pep_compass/models/apex/APEX_pathogen_models/APEX_*"):
                model = torch.load(
                    a_model,
                    map_location=torch.device(self.device),
                    weights_only=False,
                )
                model.eval()
                self.APEX_models.append(model)
        elif path == "all":
            for a_model in glob.glob(
                f"{self.file_dir}/../src/pep_compass/models/apex/Full_APEX_pathogen_models/trained_*"
            ):
                model = torch.load(
                    a_model,
                    map_location=torch.device(self.device),
                    weights_only=False,
                )
                model.eval()
                self.APEX_models.append(model)

        self.batch_size = batch_size  # change according to your GPU memory

    # Use pretrained APEX models to predict species-specific antimicrobial activity (i.e., minimum inhibitory concentration [MIC]; unit: uM)
    # 8 pretrained APEX models are provided, and predictions are averaged
    def predict(self, seq_list, use_tqdm: bool = False):
        AMP_sum = 0

        data_len = len(seq_list)
        num_models = len(self.APEX_models)
        outer_bar = tqdm(total=num_models, desc="Models") if use_tqdm else None

        for ensemble_id in range(num_models):
            AMP_model = self.APEX_models[ensemble_id].to(self.device).eval()

            inner_bar = (
                tqdm(
                    total=data_len,
                    desc=f"Sequences [{ensemble_id + 1}/{num_models}]",
                    leave=False,
                )
                if use_tqdm
                else None
            )

            batch_iter = range(int(math.ceil(data_len / float(self.batch_size))))
            for i in batch_iter:
                seq_batch = seq_list[i * self.batch_size : (i + 1) * self.batch_size]
                seq_rep = onehot_encoding(
                    seq_batch, self.max_len, self.word2idx
                )  # make input
                X_seq = torch.LongTensor(seq_rep).to(self.device)

                AMP_pred_batch = (
                    AMP_model(X_seq).cpu().detach().numpy()
                )  # make predictions
                AMP_pred_batch = (
                    10 ** (6 - AMP_pred_batch)
                )  # transform back to MICs; When training the APEX models, MICs were transformed by: -np.log10(MICs/float(1000000))

                if i == 0:
                    AMP_pred = AMP_pred_batch
                else:
                    AMP_pred = np.vstack([AMP_pred, AMP_pred_batch])

                if inner_bar is not None:
                    inner_bar.update(len(seq_batch))

            # sum up the predictions made by different APEX models
            if ensemble_id == 0:
                AMP_sum = AMP_pred
            else:
                AMP_sum += AMP_pred

            if inner_bar is not None:
                inner_bar.close()
            if outer_bar is not None:
                outer_bar.update(1)

        if outer_bar is not None:
            outer_bar.close()

        AMP_pred = AMP_sum / float(len(self.APEX_models))  # average the predictions

        return AMP_pred

class APEXBlackBox(AbstractBlackBox):
    def __init__(
        self,
        *,
        mic_aggregate: str = "mean",
        mic_bacteria: str | list = "all",
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

        self.apex_predictor = PredictorAPEX(device=device)

        if mic_aggregate == "max":
            mic_aggregate_func = lambda x: np.max(x, axis=1)
        elif mic_aggregate == "mean":
            mic_aggregate_func = lambda x: np.mean(x, axis=1)

        if mic_bacteria == "all":
            mic_bacteria_func = lambda x: x
        elif isinstance(mic_bacteria, list):
            mic_bacteria_func = lambda x: x[:, mic_bacteria]

        self.peptide_scorer = lambda x: mic_aggregate_func(
            np.log2(mic_bacteria_func(self.apex_predictor.predict(x)))
        )

        self.maximize = False

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="APEX",
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




class NegativeBlackBox(AbstractBlackBox):
    """A wrapper for a black box that negates the objective function.

    If you construct a black-box function f for maximizing, then -f is
    a black-box function for minimizing. This class is a wrapper for
    implementing the latter.

    The only difference is that the __call__ method returns -f(x) instead
    of f(x). The _black_box method is the same as the original black box.
    """

    def __init__(self, f: AbstractBlackBox):
        self.f = f
        super().__init__(
            batch_size=f.batch_size,
            parallelize=f.parallelize,
            num_workers=f.num_workers,
            evaluation_budget=cast(int | None, f.evaluation_budget),
        )

    def __call__(self, x: NDArray[np.str_], context=None):
        return -self.f.__call__(x, context)

    def _black_box(self, x: NDArray[np.str_], context=None):
        return self.f._black_box(x, context)

    def __str__(self) -> str:
        return f"NegativeBlackBox({self.f})"

    def __repr__(self) -> str:
        return f"<NegativeBlackBox({self.f})>"
    
    def get_black_box_info(self) -> BlackBoxInformation:
        return self.f.get_black_box_info()

@dataclass
class CSVObserverInitInfo:
    """Initialization information for the LoggingObserver."""

    experiment_id: str
    results_path: str | Path


class CSVObserver(AbstractObserver):
    """
    A simple observer that logs to a CSV file, appending rows on each query.
    """

    def __init__(self, maximize: bool):
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


black_box = APEXBlackBox(
    mic_aggregate="mean",
    mic_bacteria=[1, 2, 3],
    device=DEVICE,
)

observer = CSVObserver(black_box.maximize)
black_box.set_observer(observer)

negated_blackbox = NegativeBlackBox(black_box)


proteins = {
    # "middle-1": ("FLYKWWIRIGRLKL", 5),
    "jurand-4": ("KYCRRFRWLTFRWL", 5),
    # "jurand-2": ("KFRNRHRWKFKLIFRN", 5),
    # "jurand-7": ("KKYWLIRKWIRLWFLT", 5),
    # "mammuthusin-3": ("KTLKIIRLLF", 5),
    # "hydrodamin-2": ("RMARNLVRYVQGLKKKKVI", 5),
}

for name, (sequence, num) in proteins.items():
    for i in range(num):
        rng_seed = int(time())  # Create a unique rng_seed for each iteration
        observer.initialize_observer(
            negated_blackbox.get_black_box_info(),
            {
                "experiment_id": f"{sequence}_{rng_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "experiment_path": OUTPUT_PATH,
            },
            rng_seed,
        )

        x0 = np.array([list(sequence)], dtype="<U1")

        solver = LaMBO2(
            black_box=negated_blackbox,
            x0=x0,
        )

        result = solver.solve(max_iter=10)

        all_y = np.concatenate(solver.history_for_training["y"], axis=0)
        all_x = np.concatenate(solver.history_for_training["x"], axis=0)
