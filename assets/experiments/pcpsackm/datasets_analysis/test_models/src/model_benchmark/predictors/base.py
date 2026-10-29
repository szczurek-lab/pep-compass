"""Batch prediction interface shared by every published half-life model.

Each adapter exposes the same contract as the oracles of the main library:
a batch of peptides enters as one array, one numeric column leaves. The
canonical internal unit is ``log10`` seconds, because every regression metric,
the duration classes and the interval estimator of this project part operate
on that scale.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

PADDING_TOKEN = " "


@dataclass(frozen=True)
class PredictorMetadata:
    """Provenance of one published predictor.

    :param str name: Identifier used in result tables and file names.
    :param str publication: Source publication of the trained artifact.
    :param str artifact: Origin of the weights actually loaded.
    :param str representation: Input representation seen by the model.
    :param str architecture: Regression head or estimator family.
    :param str training_records: Size of the published training set, as reported.
    :param str native_units: Unit of the published model output.
    :param bool supports_gradient: Whether per-residue gradients are available.
    :param dict details: Free-form adapter specifics recorded in results.
    """

    name: str
    publication: str
    artifact: str
    representation: str
    architecture: str
    training_records: str
    native_units: str
    supports_gradient: bool = False
    details: dict = field(default_factory=dict)


class HalfLifePredictor(ABC):
    """Common batch interface for a pretrained peptide half-life regressor."""

    #: Default number of peptides pushed through the model per forward pass.
    default_batch_size = 64

    def __init__(self, metadata, batch_size=None):
        """Initialize the adapter with its provenance and batching policy.

        :param PredictorMetadata metadata: Provenance of the loaded artifact.
        :param int batch_size: Peptides per forward pass; adapter default if None.
        :rtype: None
        """
        self.metadata = metadata
        self.batch_size = int(batch_size or self.default_batch_size)

    @property
    def name(self):
        """Return the predictor identifier.

        :return: Identifier used in result tables.
        :rtype: str
        """
        return self.metadata.name

    @abstractmethod
    def _predict_log10_seconds(self, sequences):
        """Predict base-10 logarithmic half-life for one batch of peptides.

        Implementations receive a list of plain amino-acid strings and return
        finite predictions in ``log10`` seconds.

        :param list[str] sequences: Peptide sequences of one batch.
        :return: Predictions in ``log10`` seconds, shape ``(B,)``.
        :rtype: numpy.ndarray
        """

    def predict_log10_seconds(self, sequences):
        """Predict base-10 logarithmic half-life for arbitrarily many peptides.

        :param sequences: Iterable of peptide sequences.
        :return: Predictions in ``log10`` seconds, shape ``(N,)``.
        :rtype: numpy.ndarray
        :raises ValueError: If the input is empty or a prediction is not finite.
        """
        items = [str(sequence).strip() for sequence in sequences]
        if not items or any(not item for item in items):
            raise ValueError("Every peptide must be a non-empty sequence string")

        # Chunked forward passes
        chunks = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            predictions = np.asarray(self._predict_log10_seconds(batch), dtype=float)  # (B,)
            if predictions.shape != (len(batch),):
                raise ValueError(
                    f"{self.name} returned shape {predictions.shape} for {len(batch)} peptides"
                )
            chunks.append(predictions)
        result = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]  # (N,)

        if not np.isfinite(result).all():
            raise ValueError(f"{self.name} produced non-finite predictions")
        return result

    def predict_seconds(self, sequences):
        """Predict half-life in seconds.

        :param sequences: Iterable of peptide sequences.
        :return: Predicted half-life in seconds, shape ``(N,)``.
        :rtype: numpy.ndarray
        """
        return 10.0 ** self.predict_log10_seconds(sequences)  # (N,)

    def predict_interval_log10_seconds(self, sequences, space=None):
        """Return the model's own predictive interval when it publishes one.

        :param sequences: Iterable of peptide sequences.
        :param str space: Adapter-specific reading of a published calibration,
            ignored by adapters that ship no interval.
        :return: Lower and upper bounds in ``log10`` seconds, shape ``(N, 2)``,
            or None when the published artifact carries no interval.
        :rtype: numpy.ndarray | None
        """
        return None

    def __call__(self, x, context=None):
        """Score a padded token matrix, matching the library's oracle calling convention.

        :param numpy.ndarray x: Token matrix of shape ``(B, L)`` or a sequence list.
        :param dict context: Accepted for interface compatibility; unused.
        :return: Predictions in ``log10`` seconds, shape ``(B, 1)``.
        :rtype: numpy.ndarray
        """
        sequences = decode_token_matrix(x)
        return self.predict_log10_seconds(sequences).reshape(-1, 1)  # (B, 1)

    def unload(self):
        """Release any resources the adapter holds.

        Adapters that load weights override this; the default is a no-op so
        that an evaluation run can release every predictor uniformly.

        :rtype: None
        """

    def per_residue_gradient(self, sequence):
        """Return d(prediction)/d(residue representation) for one peptide.

        :param str sequence: Peptide sequence.
        :return: Gradient magnitude per residue, shape ``(L,)``.
        :rtype: numpy.ndarray
        :raises NotImplementedError: If the adapter is not differentiable.
        """
        raise NotImplementedError(f"{self.name} does not expose input gradients")

    def describe(self):
        """Return the provenance record stored next to every result table.

        :return: Serializable metadata of the loaded artifact.
        :rtype: dict
        """
        record = {
            "name": self.metadata.name,
            "publication": self.metadata.publication,
            "artifact": self.metadata.artifact,
            "representation": self.metadata.representation,
            "architecture": self.metadata.architecture,
            "training_records": self.metadata.training_records,
            "native_units": self.metadata.native_units,
            "supports_gradient": self.metadata.supports_gradient,
            "batch_size": self.batch_size,
        }
        record.update(self.metadata.details)
        return record


def decode_token_matrix(x):
    """Convert a padded token matrix into plain sequence strings.

    :param x: Token matrix of shape ``(B, L)``, or an iterable of sequences.
    :return: Peptide sequences without padding.
    :rtype: list[str]
    :raises ValueError: If the input is neither a token matrix nor sequences.
    """
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return ["".join(row).replace(PADDING_TOKEN, "") for row in x.astype(str)]
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(item).strip() for item in np.asarray(x, dtype=str).ravel()]
    raise ValueError("Expected a (B, L) token matrix or an iterable of sequences")
