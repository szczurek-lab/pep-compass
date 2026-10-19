"""Batched inference for named APEX ensembles."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from pep_compass.steps.oracles.strategies.apex.artifacts import (
    resolve_apex_weights,
)
from pep_compass.utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class _APEXUnpickler(pickle.Unpickler):
    """Resolve historical module names stored inside distributed checkpoints."""

    def find_class(self, module: str, name: str) -> object:
        """Find class.

        :param module: Module.
        :type module: str
        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :return: Result of this operation.
        :rtype: object
        """
        if module in {"APEX_models", "AMP_DL_model_twohead"}:
            module = "pep_compass.steps.oracles.strategies.apex_original.APEX_models"
        return super().find_class(module, name)


class _APEXPickleModule:
    """Expose the custom unpickler through the ``torch.load`` pickle API."""

    Unpickler = _APEXUnpickler

    def __getattr__(self, name: str) -> object:
        """Getattr.

        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :return: Result of this operation.
        :rtype: object
        """
        return getattr(pickle, name)


class APEXPredictor:
    """Load a validated ensemble once and perform batched inference.

    :param device: Torch inference device.
    :param batch_size: Maximum number of sequences per model forward call.
    :param model: Named model variant.
    :param models_directory: Optional external weight root.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_size: int = 3000,
        model: str = "default",
        models_directory: str | Path | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param device: Computation device accepted by the numerical backend.
        :type device: str
        :param batch_size: Batch size.
        :type batch_size: int
        :param model: Named model variant.
        :type model: str
        :param models_directory: Optional external model root replacing packaged artifacts.
        :type models_directory: str | Path | None
        :rtype: None
        """
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("APEX batch_size must be a positive integer.")
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model_name = model
        descriptor, paths = resolve_apex_weights(model, models_directory)
        self.pathogen_list = descriptor.pathogens
        self.max_length = 52
        self.word_to_index = _amino_acid_vocabulary()

        # Model initialization
        ## Checkpoints are loaded once and remain on the configured inference device.
        started_at = perf_counter()
        self.models = []
        for path in paths:
            loaded = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
                pickle_module=_APEXPickleModule,
            )
            self.models.append(loaded.to(self.device).eval())
        logger.info(
            "Loaded APEX ensemble model=%s models=%s device=%s duration_seconds=%.6f.",
            model,
            len(self.models),
            self.device,
            perf_counter() - started_at,
        )

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        """Return ensemble-mean MIC predictions shaped ``(B, P)``.

        :param sequences: Peptide sequences in input order.
        :type sequences: Sequence[str]
        :return: Result of this operation.
        :rtype: np.ndarray
        """
        if not sequences:
            return np.empty((0, len(self.pathogen_list)), dtype=np.float32)
        model_predictions = []
        with torch.inference_mode():
            for model in self.models:
                batches = []
                for start in range(0, len(sequences), self.batch_size):
                    encoded = _encode_sequences(
                        sequences[start : start + self.batch_size],
                        self.max_length,
                        self.word_to_index,
                    )
                    inputs = torch.as_tensor(
                        encoded,
                        dtype=torch.long,
                        device=self.device,
                    )  # (B_model, L)
                    transformed = model(inputs)  # (B_model, P)
                    mic = torch.pow(10.0, 6.0 - transformed)  # (B_model, P)
                    batches.append(mic.cpu().numpy())
                model_predictions.append(np.concatenate(batches, axis=0))
        return np.mean(np.stack(model_predictions, axis=0), axis=0)  # (B, P)


def _amino_acid_vocabulary() -> dict[str, int]:
    """Return the integer vocabulary used when the APEX models were trained.

    :return: Result of this operation.
    :rtype: dict[str, int]
    """
    symbols = ("0", "1", "2", *tuple("ACDEFGHIKLMNPQRSTVWY"))
    return {symbol: index for index, symbol in enumerate(symbols)}


def _encode_sequences(
    sequences: Sequence[str],
    max_length: int,
    vocabulary: dict[str, int],
) -> np.ndarray:
    """Encode peptide strings with historical start/end tokens.

    :return: Integer token matrix shaped ``(B, L)``.

    :param sequences: Peptide sequences in input order.
    :type sequences: Sequence[str]
    :param max_length: Max length.
    :type max_length: int
    :param vocabulary: Vocabulary.
    :type vocabulary: dict[str, int]
    :rtype: np.ndarray
    """
    encoded = np.zeros((len(sequences), max_length), dtype=np.int64)  # (B, L)
    for row, sequence in enumerate(sequences):
        framed = "1" + sequence[: max_length - 2].upper() + "2"
        encoded[row, : len(framed)] = [vocabulary.get(symbol, 0) for symbol in framed]
    return encoded
