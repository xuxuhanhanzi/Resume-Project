"""A small, fully inspectable Decoder-only language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from forgellm.model.attention import CausalSelfAttention, KVCache
from forgellm.model.config import ModelConfig
from forgellm.model.layers import RMSNorm, SwiGLU


@dataclass(frozen=True, slots=True)
class DecoderOutput:
    """Language-model logits plus optional training/inference internals."""

    logits: Tensor
    cache: tuple[KVCache, ...] | None = None
    hidden_states: Tensor | None = None


class TransformerBlock(nn.Module):
    """Pre-norm attention and SwiGLU sublayers with residual connections."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = SwiGLU(config.d_model, config.ffn_hidden_dim)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        """Apply both residual sublayers."""
        attended, next_cache = self.attention(
            self.attention_norm(hidden_states), cache=cache, use_cache=use_cache
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states, next_cache


class DecoderLM(nn.Module):
    """Decoder-only causal language model with optional embedding weight tying."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_module)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: tuple[KVCache, ...] | None = None,
        use_cache: bool = False,
        return_hidden_states: bool = False,
    ) -> DecoderOutput:
        """Return next-token logits for every supplied token position."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids must contain at least one token")
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("cache must contain exactly one KVCache per decoder block")

        hidden_states = self.token_embedding(input_ids)
        next_caches: list[KVCache] = []
        for layer_index, block in enumerate(self.blocks):
            layer_cache = cache[layer_index] if cache is not None else None
            hidden_states, next_cache = block(
                hidden_states,
                cache=layer_cache,
                use_cache=use_cache,
            )
            if next_cache is not None:
                next_caches.append(next_cache)

        normalized_hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(normalized_hidden_states)
        resolved_cache = tuple(next_caches) if use_cache else None
        exposed_hidden_states = normalized_hidden_states if return_hidden_states else None
        return DecoderOutput(
            logits=logits,
            cache=resolved_cache,
            hidden_states=exposed_hidden_states,
        )

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        """Count unique parameters, avoiding double-counting tied embeddings."""
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)


def next_token_loss(logits: Tensor, input_ids: Tensor, *, ignore_index: int = -100) -> Tensor:
    """Compute teacher-forced next-token cross entropy with the required one-token shift."""
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [B, T, V] and input_ids must be [B, T]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits batch/sequence dimensions must match input_ids")
    if input_ids.size(1) < 2:
        raise ValueError("next-token loss requires at least two token positions")
    predictions = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        predictions.view(-1, predictions.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
    )
