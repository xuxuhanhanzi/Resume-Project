"""Causal self-attention with a readable MHA/GQA reference path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from forgellm.model.config import ModelConfig
from forgellm.model.layers import RMSNorm, RotaryEmbedding


@dataclass(frozen=True, slots=True)
class KVCache:
    """Unexpanded rotary key/value tensors stored for autoregressive decoding."""

    key: Tensor
    value: Tensor

    def __post_init__(self) -> None:
        if self.key.ndim != 4 or self.value.ndim != 4:
            raise ValueError("cached key/value must have shape [B, H_kv, T, head_dim]")
        if self.key.shape != self.value.shape:
            raise ValueError("cached key and value must have identical shapes")

    @property
    def sequence_length(self) -> int:
        """Number of cached token positions."""
        return self.key.size(-2)


def repeat_kv(inputs: Tensor, repeats: int) -> Tensor:
    """Expand KV heads so each one is shared by ``repeats`` query heads."""
    if inputs.ndim != 4:
        raise ValueError("inputs must have shape [B, H_kv, T, head_dim]")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if repeats == 1:
        return inputs
    batch, kv_heads, sequence, head_dim = inputs.shape
    expanded = inputs[:, :, None, :, :].expand(batch, kv_heads, repeats, sequence, head_dim)
    return expanded.reshape(batch, kv_heads * repeats, sequence, head_dim)


def causal_attention_mask(
    query_length: int,
    key_length: int,
    *,
    past_length: int,
    device: torch.device,
) -> Tensor:
    """Return a boolean mask where ``True`` means a key is visible to a query."""
    if query_length <= 0 or key_length <= 0:
        raise ValueError("query_length and key_length must be positive")
    if past_length < 0 or key_length != past_length + query_length:
        raise ValueError("key_length must equal past_length + query_length")
    query_positions = torch.arange(
        past_length, past_length + query_length, device=device
    ).unsqueeze(-1)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    return key_positions <= query_positions


def manual_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    allowed_mask: Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tensor:
    """Readable attention equation used as the project's numerical oracle."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must have shape [B, H, T, head_dim]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[:2] != key.shape[:2] or query.size(-1) != key.size(-1):
        raise ValueError("query and key batch/head/head_dim dimensions must match")
    expected_mask_shape = (query.size(-2), key.size(-2))
    if allowed_mask.shape != expected_mask_shape or allowed_mask.dtype != torch.bool:
        raise ValueError(f"allowed_mask must be bool with shape {expected_mask_shape}")

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    scores = scores.masked_fill(~allowed_mask, float("-inf"))
    probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
    probabilities = F.dropout(probabilities, p=dropout_p, training=training)
    return torch.matmul(probabilities, value)


class CausalSelfAttention(nn.Module):
    """Pre-RoPE causal attention supporting MHA, GQA, and an inference KV cache."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

    def _split_heads(self, projected: Tensor, head_count: int) -> Tensor:
        batch, sequence, _ = projected.shape
        return projected.view(batch, sequence, head_count, self.config.head_dim).transpose(1, 2)

    def _validate_cache(self, cache: KVCache, batch_size: int) -> None:
        expected = (batch_size, self.config.n_kv_heads, cache.sequence_length, self.config.head_dim)
        if cache.key.shape != expected:
            raise ValueError(f"cache shape must be {expected}, received {tuple(cache.key.shape)}")
        if cache.key.device != self.q_proj.weight.device:
            raise ValueError("cache and attention parameters must use the same device")

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        """Apply causal attention and optionally return the appended KV cache."""
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.config.d_model:
            raise ValueError("hidden_states must have shape [B, T, d_model]")
        batch, query_length, _ = hidden_states.shape
        if query_length == 0:
            raise ValueError("attention does not accept an empty sequence")

        past_length = 0
        if cache is not None:
            self._validate_cache(cache, batch)
            past_length = cache.sequence_length
        if past_length + query_length > self.config.max_seq_len:
            raise ValueError("attention sequence exceeds max_seq_len")

        query = self._split_heads(self.q_proj(hidden_states), self.config.n_heads)
        key = self._split_heads(self.k_proj(hidden_states), self.config.n_kv_heads)
        value = self._split_heads(self.v_proj(hidden_states), self.config.n_kv_heads)
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        positions = torch.arange(
            past_length,
            past_length + query_length,
            device=hidden_states.device,
        )
        query, key = self.rope(query, key, positions)

        if cache is not None:
            key = torch.cat((cache.key, key), dim=-2)
            value = torch.cat((cache.value, value), dim=-2)
        next_cache = KVCache(key=key, value=value) if use_cache else None

        expanded_key = repeat_kv(key, self.config.queries_per_kv)
        expanded_value = repeat_kv(value, self.config.queries_per_kv)
        allowed_mask = causal_attention_mask(
            query_length,
            key.size(-2),
            past_length=past_length,
            device=hidden_states.device,
        )
        if self.config.attention_backend == "manual":
            attended = manual_scaled_dot_product_attention(
                query,
                expanded_key,
                expanded_value,
                allowed_mask=allowed_mask,
                dropout_p=self.config.dropout,
                training=self.training,
            )
        else:
            attended = F.scaled_dot_product_attention(
                query,
                expanded_key,
                expanded_value,
                attn_mask=allowed_mask,
                dropout_p=self.config.dropout if self.training else 0.0,
            )

        merged = attended.transpose(1, 2).contiguous().view(batch, query_length, -1)
        return self.out_proj(merged), next_cache
