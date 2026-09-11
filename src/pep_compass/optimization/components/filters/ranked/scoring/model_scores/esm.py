"""ESM pseudo-log-likelihood scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pep_compass.optimization.components.filters.ranked.scoring.base import ScoreFunction


@dataclass(frozen=True)
class ESM2ModelSpec:
    name: str
    pretrained_fn: str


ESM2_MODEL_SPECS = {
    "esm2_t6_8M_UR50D": ESM2ModelSpec(
        name="esm2_t6_8M_UR50D", pretrained_fn="esm2_t6_8M_UR50D"
    ),
    "esm2_t12_35M_UR50D": ESM2ModelSpec(
        name="esm2_t12_35M_UR50D", pretrained_fn="esm2_t12_35M_UR50D"
    ),
    "esm2_t30_150M_UR50D": ESM2ModelSpec(
        name="esm2_t30_150M_UR50D", pretrained_fn="esm2_t30_150M_UR50D"
    ),
    "esm2_t33_650M_UR50D": ESM2ModelSpec(
        name="esm2_t33_650M_UR50D", pretrained_fn="esm2_t33_650M_UR50D"
    ),
    "esm2_t36_3B_UR50D": ESM2ModelSpec(
        name="esm2_t36_3B_UR50D", pretrained_fn="esm2_t36_3B_UR50D"
    ),
}


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
        if model_name not in ESM2_MODEL_SPECS:
            available = ", ".join(sorted(ESM2_MODEL_SPECS))
            raise ValueError(
                f"Unsupported ESM2 model '{model_name}'. Available: {available}"
            )

        try:
            import esm
        except ImportError as exc:
            raise ImportError(
                "ESM package not found. Install with: uv pip install fair-esm"
            ) from exc

        self.model_name = model_name
        self.device = device

        pretrained_fn_name = ESM2_MODEL_SPECS[model_name].pretrained_fn
        pretrained_fn = getattr(esm.pretrained, pretrained_fn_name, None)
        if pretrained_fn is None:
            raise ValueError(
                f"Model '{model_name}' is unavailable in installed fair-esm version."
            )

        model, alphabet = pretrained_fn()
        self.model = model.eval().to(device)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

        self.pad_idx = alphabet.padding_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx

    @torch.no_grad()
    def pll(self, sequence: str) -> float:
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
        return self.pll(sequence) >= threshold

    def __call__(self, batch, context) -> torch.Tensor:
        """Return ESM pseudo-log-likelihood scores aligned with the batch."""
        del context
        return torch.as_tensor(
            [self.pll(sequence) for sequence in batch.sequences],
            dtype=batch.latent_origins.dtype,
            device=batch.latent_origins.device,
        )
