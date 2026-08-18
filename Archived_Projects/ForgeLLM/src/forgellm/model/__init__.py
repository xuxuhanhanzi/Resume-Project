"""Educational Decoder-only Transformer and modern architecture laboratories."""

from forgellm.model.attention import CausalSelfAttention, KVCache
from forgellm.model.config import ModelConfig, ModelConfigError
from forgellm.model.decoder import DecoderLM, DecoderOutput, next_token_loss
from forgellm.model.frontier_attention import (
    CompressedHybridAttentionLite,
    GatedDeltaNet,
    HybridMixerStack,
    MultiHeadLatentAttention,
    ScaledRotaryEmbedding,
    moba_reference_attention,
)
from forgellm.model.frontier_layers import (
    AttentionResiduals,
    ManifoldHyperConnectionLite,
    MultiTokenPredictionHead,
    SparseMoE,
)
from forgellm.model.generation import GenerationConfig, generate
from forgellm.model.layers import RMSNorm, RotaryEmbedding, SwiGLU
from forgellm.model.optim import Muon
from forgellm.model.systems import (
    Int8WeightOnlyLinear,
    online_softmax_blockwise_attention,
    quantize_symmetric_int8,
)

__all__ = [
    "CausalSelfAttention",
    "CompressedHybridAttentionLite",
    "AttentionResiduals",
    "DecoderLM",
    "DecoderOutput",
    "GenerationConfig",
    "GatedDeltaNet",
    "HybridMixerStack",
    "Int8WeightOnlyLinear",
    "KVCache",
    "ModelConfig",
    "ModelConfigError",
    "ManifoldHyperConnectionLite",
    "MultiHeadLatentAttention",
    "MultiTokenPredictionHead",
    "Muon",
    "RMSNorm",
    "RotaryEmbedding",
    "ScaledRotaryEmbedding",
    "SparseMoE",
    "SwiGLU",
    "generate",
    "moba_reference_attention",
    "next_token_loss",
    "online_softmax_blockwise_attention",
    "quantize_symmetric_int8",
]
