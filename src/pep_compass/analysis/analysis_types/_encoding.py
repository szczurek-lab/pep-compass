"""Shared HydrAMP encoding helper for analyses that need fresh latent positions."""

from __future__ import annotations

import numpy as np
import torch

from pep_compass.autoencoder.strategies.hydramp.adapter import HydrampAutoencoder


def encode_sequences(
    autoencoder: HydrampAutoencoder,
    sequences: list[str],
    batch_size: int,
    progress=None,
) -> np.ndarray:
    """Encode peptide sequences into their latent means, batched.

    :param autoencoder: A constructed, ``eval()``-mode ``HydrampAutoencoder``.
    :param sequences: Peptide strings to encode.
    :param batch_size: Sequences per forward pass.
    :param progress: Optional object exposing ``update(rows: int)``, e.g. a
        :class:`~pep_compass.analysis.analysis_types.locality._streaming.ProgressReporter`
        or ``NestedProgress``.
    :return: Latent means shaped ``(len(sequences), D)``.
    """
    vectors = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            vectors.append(autoencoder.encode_peptides(batch).detach().cpu().numpy())
            if progress is not None:
                progress.update(len(batch))
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 0))
