"""ESM pseudo-log-likelihood scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pep_compass.steps.filters.strategies.ranked.scoring.base import ScoreFunction

if TYPE_CHECKING:
    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.data_schemas.candidates import CandidateBatch
    from pep_compass.optimization.engine.execution.context import OptimizationContext


ESM2_MODEL_NAMES = (
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
)


class ESM2PPLScorer(ScoreFunction):
    """
    Scores sequence plausibility with ESM2 using average pseudo log-likelihood.
    Higher is better (less negative).
    """

    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        device: str = "cpu",
    ) -> None:
        """Initialize this object with the declared configuration.

        :param model_name: Named model variant.
        :type model_name: str
        :param device: Computation device accepted by the numerical backend.
        :type device: str
        :rtype: None
        """
        self._configure({"model_name": model_name, "device": device})
        if model_name not in ESM2_MODEL_NAMES:
            available = ", ".join(sorted(ESM2_MODEL_NAMES))
            raise ValueError(
                f"Unsupported ESM2 model '{model_name}'. Available: {available}"
            )

        self.model_name, self.device = model_name, device

    def _initialize(self, autoencoder: Autoencoder | None = None) -> None:
        """Initialize.

        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: Autoencoder | None
        :rtype: None
        """
        try:
            import esm
        except ImportError as exc:
            raise ImportError(
                "ESM package not found. Install project dependencies with: "
                "uv sync --extra cpu (or a supported CUDA extra)"
            ) from exc

        pretrained_fn_name = self.model_name
        pretrained_fn = getattr(esm.pretrained, pretrained_fn_name, None)
        if pretrained_fn is None:
            raise ValueError(
                f"Model '{self.model_name}' is unavailable in installed fair-esm version."
            )

        model, alphabet = pretrained_fn()
        self.model = model.eval().to(self.device)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

        self.pad_idx = alphabet.padding_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx

    @torch.no_grad()
    def pll(self, sequence: str) -> float:
        """Pll.

        :param sequence: One peptide sequence.
        :type sequence: str
        :return: Result of this operation.
        :rtype: float
        """
        if not sequence:
            raise ValueError("Sequence cannot be empty.")

        _, _, batch_tokens = self.batch_converter([("seq", sequence)])
        batch_tokens = batch_tokens.to(self.device)

        out = self.model(batch_tokens)
        log_probs = torch.log_softmax(out["logits"], dim=-1)
        tokens = batch_tokens[0]
        token_log_probs = log_probs[0]

        valid_mask = (
            (tokens != self.pad_idx)
            & (tokens != self.cls_idx)
            & (tokens != self.eos_idx)
        )

        aa_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if aa_positions.numel() == 0:
            raise ValueError("Sequence contains no valid amino-acid tokens.")

        selected = token_log_probs[aa_positions, tokens[aa_positions]]
        return float(selected.mean().item())

    def passes_threshold(self, sequence: str, threshold: float) -> bool:
        """Passes threshold.

        :param sequence: One peptide sequence.
        :type sequence: str
        :param threshold: Threshold.
        :type threshold: float
        :return: Result of this operation.
        :rtype: bool
        """
        return self.pll(sequence) >= threshold

    def __call__(
        self, batch: CandidateBatch, context: OptimizationContext
    ) -> torch.Tensor:
        """Return ESM pseudo-log-likelihood scores aligned with the batch.

        :param batch: Incoming aligned candidate batch.
        :type batch: CandidateBatch
        :param context: Scoped execution services, shared run state and random generator.
        :type context: OptimizationContext
        :return: Result of this operation.
        :rtype: torch.Tensor
        """
        del context
        return torch.as_tensor(
            [self.pll(sequence) for sequence in batch.sequences],
            dtype=batch.latent_origins.dtype,
            device=batch.latent_origins.device,
        )
