import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation
import torch.nn.functional as F
from einops import rearrange


_battleamp_predictor = None
_battleamp_predictor_key = None


def _predict_in_isolated_process(
    sequences: list[str],
    model: str,
    models_directory: str | None,
) -> np.ndarray:
    """Evaluate BattleAMP in the reusable TensorFlow worker process."""
    global _battleamp_predictor, _battleamp_predictor_key
    predictor_key = (model, models_directory)
    if _battleamp_predictor is None or _battleamp_predictor_key != predictor_key:
        from pep_compass.optimization.components.oracles.strategies.battleamp.BattleAMPPredictor import (
            PredictorBattleAMP,
        )

        _battleamp_predictor = PredictorBattleAMP(
            device="cpu",
            model=model,
            models_directory=models_directory,
        )
        _battleamp_predictor_key = predictor_key
    return _battleamp_predictor.predict(sequences).flatten()


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
        
        self.device = device
        self.model_name = model
        self.models_directory = models_directory
        self._executor = None
        if str(device).startswith("cuda"):
            # Max - debbuging: Zmiana z uruchamiania BattleAMP i PyTorch w jednym procesie na uruchamianie BattleAMP w osobnym procesie CPU ~TensorFlow pozostaje odseparowany od procesu, w którym PyTorch wykonuje LE-BO na GPU, dzięki czemu nie inicjalizuje tam CUDA i nie zakłóca działania cuSOLVER.
            self._executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=get_context("spawn"),
            )
            self.battleamp_predictor = None
        else:
            from pep_compass.optimization.components.oracles.strategies.battleamp.BattleAMPPredictor import (
                PredictorBattleAMP,
            )

            self.battleamp_predictor = PredictorBattleAMP(
                device=device,
                model=model,
                models_directory=models_directory,
            )

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
        if self._executor is not None:
            predictions = self._executor.submit(
                _predict_in_isolated_process,
                sequences,
                self.model_name,
                self.models_directory,
            ).result()
        else:
            predictions = self.battleamp_predictor.predict(sequences).flatten()
        predictions = np.log2(predictions)

        for i, seq in enumerate(sequences):
            self.cache.append((seq, predictions[i].item()))

        return predictions.reshape(-1, 1)
    
    def clear_cache(self):
        self.cache = []

    def terminate(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        super().terminate()
