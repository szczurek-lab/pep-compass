from .MBC_Attention_Predictor import PredictorMBCAttention

__all__ = ["PredictorMBCAttention"]
"""MBC-Attention oracle strategy and bundled reference implementation."""

from pep_compass.optimization.components.oracles.strategies.mbc_attention.oracle import (
    MBCAttentionBlackBox,
)

__all__ = ["MBCAttentionBlackBox"]
