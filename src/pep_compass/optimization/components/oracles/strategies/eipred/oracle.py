import numpy as np
from poli_baselines.core.abstract_solver import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation

from pep_compass.optimization.components.oracles.strategies.eipred.eippred import EIPredPredictor

class EIPredBlackBox(AbstractBlackBox):
    def __init__(
        self,
        *,
        mic_aggregate: str = "mean",
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
        model: str = "default",
        models_directory: str | None = None,
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )

        # Model-backed predictor
        ## Resolve both classifier weights and the selected-feature schema
        self.predictor = EIPredPredictor(
            device="cpu",
            batch_size=batch_size or 1000,
            model=model,
            models_directory=models_directory,
        )

        if mic_aggregate == "max":
            mic_aggregate_func = lambda x: np.max(x, axis=1)
        elif mic_aggregate == "mean":
            mic_aggregate_func = lambda x: np.mean(x, axis=1)
        else:
            raise ValueError("mic_aggregate must be 'mean' or 'max'")

        # EIPred model outputs 10**(-y_pred), apply log2 transformation
        self.peptide_scorer = lambda seqs: mic_aggregate_func(np.log2(self.predictor.predict(seqs)).reshape(-1, 1))

        self.cache = []
        self.shift = 0.0

    def set_shift(self, shift: float):
        self.shift = shift

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="EIPred",
            max_sequence_length=50,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=list("ACDEFGHIKLMNPQRSTVWY"),
            log_transform_recommended=False,
            discrete=True,
            padding_token=" ",
        )

    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        # x expected as array-like of sequences (list of chars per sequence)
        sequences = ["".join(seq) for seq in x]
        predictions = self.peptide_scorer(sequences)

        if context is not None and isinstance(context, dict):
            context['sequences'] = sequences

        for i, seq in enumerate(sequences):
            self.cache.append((seq, float(predictions[i])))

        return predictions.reshape(-1, 1) + self.shift

    def clear_cache(self):
        self.cache = []
