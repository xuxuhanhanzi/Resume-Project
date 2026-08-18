"""Small reference implementations of modern long-context attention ideas.

These modules teach the algorithmic contracts behind recent open-weight models.
They are intentionally readable PyTorch references, not production kernels and
not claims of exact end-to-end reproduction of any vendor model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn

from forgellm.model.attention import causal_attention_mask, manual_scaled_dot_product_attention
from forgellm.model.layers import RotaryEmbedding, rotate_half

RoPEScalingMode = Literal["none", "linear", "dynamic_ntk"]


class ScaledRotaryEmbedding(nn.Module):
    """RoPE with explicit linear or dynamic-NTK context extension policies."""

    inv_freq: Tensor

    def __init__(
        self,
        head_dim: int,
        *,
        original_max_seq_len: int,
        max_seq_len: int,
        theta: float = 10_000.0,
        scaling_factor: float = 1.0,
        mode: RoPEScalingMode = "none",
    ) -> None:
        super().__init__()
        if head_dim <= 2 or head_dim % 2 != 0:
            raise ValueError("head_dim must be even and greater than two")
        if original_max_seq_len <= 0 or max_seq_len < original_max_seq_len:
            raise ValueError("max_seq_len must be at least original_max_seq_len > 0")
        if theta <= 0 or scaling_factor < 1.0:
            raise ValueError("theta must be positive and scaling_factor must be at least one")
        if mode not in ("none", "linear", "dynamic_ntk"):
            raise ValueError("unsupported RoPE scaling mode")
        self.head_dim = head_dim
        self.original_max_seq_len = original_max_seq_len
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling_factor = scaling_factor
        self.mode = mode
        self.register_buffer("inv_freq", self._frequencies(theta), persistent=False)

    def _frequencies(self, theta: float) -> Tensor:
        frequencies: Tensor = 1.0 / (
            theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        return frequencies

    def _scaled_inputs(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        if positions.ndim != 1:
            raise ValueError("positions must be rank one")
        if positions.numel() and (positions.min() < 0 or positions.max() >= self.max_seq_len):
            raise ValueError(f"positions must be in [0, {self.max_seq_len})")
        if self.mode == "linear":
            return positions.float() / self.scaling_factor, self.inv_freq
        if self.mode == "dynamic_ntk" and positions.numel():
            sequence_length = int(positions.max()) + 1
            if sequence_length > self.original_max_seq_len:
                stretch = self.scaling_factor * sequence_length / self.original_max_seq_len - (
                    self.scaling_factor - 1.0
                )
                adjusted_theta = self.theta * stretch ** (self.head_dim / (self.head_dim - 2))
                return positions.float(), self._frequencies(adjusted_theta).to(positions.device)
        return positions.float(), self.inv_freq

    def angles(self, positions: Tensor, *, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Build scaled cosine/sine tables for supplied absolute positions."""
        scaled_positions, frequencies = self._scaled_inputs(positions)
        angles = torch.outer(scaled_positions, frequencies.to(positions.device))
        repeated = torch.repeat_interleave(angles, 2, dim=-1)
        return (
            repeated.cos()[None, None].to(dtype=dtype),
            repeated.sin()[None, None].to(dtype=dtype),
        )

    def forward(self, inputs: Tensor, positions: Tensor) -> Tensor:
        """Apply scaled RoPE to ``[B, H, T, head_dim]`` inputs."""
        if inputs.ndim != 4 or inputs.size(-1) != self.head_dim:
            raise ValueError("inputs must have shape [B, H, T, head_dim]")
        if inputs.size(-2) != positions.numel():
            raise ValueError("positions length must match sequence length")
        cos, sin = self.angles(positions.to(inputs.device), dtype=inputs.dtype)
        return inputs * cos + rotate_half(inputs) * sin


@dataclass(frozen=True, slots=True)
class MLACache:
    """Compressed latent KV state plus the decoupled rotary key component."""

    latent_kv: Tensor
    rotary_key: Tensor

    def __post_init__(self) -> None:
        if self.latent_kv.ndim != 3:
            raise ValueError("latent_kv must have shape [B, T, latent_dim]")
        if self.rotary_key.ndim != 4 or self.rotary_key.size(1) != 1:
            raise ValueError("rotary_key must have shape [B, 1, T, rope_dim]")
        if self.latent_kv.shape[:2] != (
            self.rotary_key.size(0),
            self.rotary_key.size(2),
        ):
            raise ValueError("MLA cache batch and sequence dimensions must agree")

    @property
    def sequence_length(self) -> int:
        return self.latent_kv.size(1)


class MultiHeadLatentAttention(nn.Module):
    """Educational MLA with compressed latent KV and decoupled shared RoPE keys.

    Unlike ordinary MHA, the cache stores one low-rank latent vector and one
    shared rotary key per token. Head-specific K/V tensors are reconstructed
    when attention is evaluated.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        q_rank: int,
        kv_rank: int,
        nope_dim: int,
        rope_dim: int,
        value_dim: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        dimensions = (d_model, n_heads, q_rank, kv_rank, nope_dim, rope_dim, value_dim)
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("all MLA dimensions must be positive")
        if rope_dim % 2 != 0:
            raise ValueError("rope_dim must be even")
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_rank = q_rank
        self.kv_rank = kv_rank
        self.nope_dim = nope_dim
        self.rope_dim = rope_dim
        self.value_dim = value_dim
        self.q_down = nn.Linear(d_model, q_rank, bias=False)
        self.q_up = nn.Linear(q_rank, n_heads * (nope_dim + rope_dim), bias=False)
        self.kv_down = nn.Linear(d_model, kv_rank, bias=False)
        self.k_up = nn.Linear(kv_rank, n_heads * nope_dim, bias=False)
        self.v_up = nn.Linear(kv_rank, n_heads * value_dim, bias=False)
        self.rotary_key_proj = nn.Linear(d_model, rope_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * value_dim, d_model, bias=False)
        self.rope = RotaryEmbedding(rope_dim, max_seq_len)

    @property
    def compressed_cache_elements_per_token(self) -> int:
        return self.kv_rank + self.rope_dim

    @property
    def mha_cache_elements_per_token(self) -> int:
        return 2 * self.n_heads * self.value_dim

    def _apply_rope(self, inputs: Tensor, positions: Tensor) -> Tensor:
        cos, sin = self.rope.angles(positions.to(inputs.device), dtype=inputs.dtype)
        return inputs * cos + rotate_half(inputs) * sin

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cache: MLACache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, MLACache | None]:
        """Apply compressed latent attention with a causal mask."""
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.d_model:
            raise ValueError("hidden_states must be [B, T, d_model]")
        batch, query_length, _ = hidden_states.shape
        if query_length == 0:
            raise ValueError("MLA does not accept an empty sequence")
        past_length = cache.sequence_length if cache is not None else 0
        positions = torch.arange(
            past_length, past_length + query_length, device=hidden_states.device
        )

        query = self.q_up(self.q_down(hidden_states)).view(
            batch, query_length, self.n_heads, self.nope_dim + self.rope_dim
        )
        query = query.transpose(1, 2)
        query_nope, query_rope = torch.split(query, [self.nope_dim, self.rope_dim], dim=-1)
        query_rope = self._apply_rope(query_rope, positions)

        current_latent = self.kv_down(hidden_states)
        current_rotary_key = self.rotary_key_proj(hidden_states).unsqueeze(1)
        current_rotary_key = self._apply_rope(current_rotary_key, positions)
        if cache is not None:
            latent = torch.cat((cache.latent_kv, current_latent), dim=1)
            rotary_key = torch.cat((cache.rotary_key, current_rotary_key), dim=2)
        else:
            latent = current_latent
            rotary_key = current_rotary_key
        next_cache = MLACache(latent, rotary_key) if use_cache else None

        key_nope = self.k_up(latent).view(batch, -1, self.n_heads, self.nope_dim)
        key_nope = key_nope.transpose(1, 2)
        expanded_rotary_key = rotary_key.expand(-1, self.n_heads, -1, -1)
        key = torch.cat((key_nope, expanded_rotary_key), dim=-1)
        query = torch.cat((query_nope, query_rope), dim=-1)
        value = self.v_up(latent).view(batch, -1, self.n_heads, self.value_dim)
        value = value.transpose(1, 2)

        mask = causal_attention_mask(
            query_length,
            latent.size(1),
            past_length=past_length,
            device=hidden_states.device,
        )
        attended = manual_scaled_dot_product_attention(query, key, value, allowed_mask=mask)
        merged = attended.transpose(1, 2).contiguous().view(batch, query_length, -1)
        return cast(Tensor, self.out_proj(merged)), next_cache


def moba_reference_mask(
    query: Tensor,
    key: Tensor,
    *,
    block_size: int,
    top_k_blocks: int,
) -> Tensor:
    """Build a causal MoBA-style block-selection mask using parameter-free scores."""
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query and key must share shape [B, H, T, D]")
    if block_size <= 0 or top_k_blocks <= 0:
        raise ValueError("block_size and top_k_blocks must be positive")
    batch, heads, sequence, _ = query.shape
    block_count = math.ceil(sequence / block_size)
    summaries = []
    for block_index in range(block_count):
        start = block_index * block_size
        end = min(sequence, start + block_size)
        summaries.append(key[:, :, start:end].mean(dim=-2))
    block_keys = torch.stack(summaries, dim=-2)
    mask = torch.zeros(batch, heads, sequence, sequence, dtype=torch.bool, device=query.device)

    for position in range(sequence):
        current_block = position // block_size
        selected = torch.full(
            (batch, heads, top_k_blocks),
            current_block,
            dtype=torch.long,
            device=query.device,
        )
        previous_count = min(top_k_blocks - 1, current_block)
        if previous_count > 0:
            scores = torch.einsum(
                "bhd,bhkd->bhk", query[:, :, position], block_keys[:, :, :current_block]
            )
            chosen_previous = torch.topk(scores, k=previous_count, dim=-1).indices
            selected[:, :, :previous_count] = chosen_previous
        for slot in range(previous_count + 1):
            chosen = selected[:, :, slot]
            for block_index in range(block_count):
                block_selected = chosen.eq(block_index)
                start = block_index * block_size
                end = min(sequence, start + block_size)
                mask[:, :, position, start:end] |= block_selected.unsqueeze(-1)
    causal = torch.ones(sequence, sequence, dtype=torch.bool, device=query.device).tril()
    return mask & causal


def moba_reference_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    block_size: int,
    top_k_blocks: int,
) -> tuple[Tensor, Tensor]:
    """Apply naive MoBA-style sparse attention and return its inspectable mask."""
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    mask = moba_reference_mask(query, key, block_size=block_size, top_k_blocks=top_k_blocks)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    probabilities = torch.softmax(scores.masked_fill(~mask, float("-inf")).float(), dim=-1)
    output = torch.matmul(probabilities.to(query.dtype), value)
    return output, mask


class GatedDeltaNet(nn.Module):
    """Readable recurrent gated delta-rule sequence mixer.

    State update for each head is
    ``S_t = decay_t S_(t-1) + beta_t (v_t - S_(t-1) k_t) k_t^T``.
    The prediction is ``S_t q_t``. This loop is an algorithm oracle; efficient
    implementations use chunked scans or fused kernels.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        self.decay_proj = nn.Linear(d_model, n_heads, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _heads(self, inputs: Tensor) -> Tensor:
        batch, sequence, _ = inputs.shape
        return inputs.view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the causal recurrence and return every output plus final state."""
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.d_model:
            raise ValueError("hidden_states must be [B, T, d_model]")
        batch, sequence, _ = hidden_states.shape
        if sequence == 0:
            raise ValueError("DeltaNet does not accept an empty sequence")
        expected_state = (batch, self.n_heads, self.head_dim, self.head_dim)
        if state is None:
            state = torch.zeros(
                expected_state, dtype=hidden_states.dtype, device=hidden_states.device
            )
        elif state.shape != expected_state:
            raise ValueError(f"state must have shape {expected_state}")

        query = torch.nn.functional.normalize(self._heads(self.q_proj(hidden_states)), dim=-1)
        key = torch.nn.functional.normalize(self._heads(self.k_proj(hidden_states)), dim=-1)
        value = self._heads(self.v_proj(hidden_states))
        beta = torch.sigmoid(self.beta_proj(hidden_states)).transpose(1, 2)
        decay = torch.sigmoid(self.decay_proj(hidden_states)).transpose(1, 2)
        outputs = []
        for position in range(sequence):
            key_t = key[:, :, position]
            query_t = query[:, :, position]
            value_t = value[:, :, position]
            prediction = torch.einsum("bhde,bhe->bhd", state, key_t)
            error = value_t - prediction
            correction = torch.einsum("bhd,bhe->bhde", error, key_t)
            state = decay[:, :, position, None, None] * state + (
                beta[:, :, position, None, None] * correction
            )
            outputs.append(torch.einsum("bhde,bhe->bhd", state, query_t))
        stacked = torch.stack(outputs, dim=2).transpose(1, 2).contiguous()
        merged = stacked.view(batch, sequence, self.d_model)
        return cast(Tensor, self.out_proj(merged)), state


def compressed_sparse_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    index_query: Tensor,
    index_key: Tensor,
    *,
    local_window: int,
    top_k: int,
) -> tuple[Tensor, Tensor]:
    """CSA-style reference: local causal keys plus compressed-index top-k keys."""
    if query.shape != key.shape or key.shape != value.shape or query.ndim != 4:
        raise ValueError("query/key/value must share shape [B, H, T, D]")
    if index_query.shape[:-1] != query.shape[:-1] or index_key.shape != index_query.shape:
        raise ValueError("compressed index tensors must align with query/key prefixes")
    if local_window <= 0 or top_k < 0:
        raise ValueError("local_window must be positive and top_k non-negative")
    batch, heads, sequence, _ = query.shape
    mask = torch.zeros(batch, heads, sequence, sequence, dtype=torch.bool, device=query.device)
    for position in range(sequence):
        local_start = max(0, position - local_window + 1)
        mask[:, :, position, local_start : position + 1] = True
        older_end = local_start
        selected_count = min(top_k, older_end)
        if selected_count:
            scores = torch.einsum(
                "bhr,bhkr->bhk",
                index_query[:, :, position],
                index_key[:, :, :older_end],
            )
            selected = torch.topk(scores, selected_count, dim=-1).indices
            mask[:, :, position].scatter_(dim=-1, index=selected, value=True)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    probabilities = torch.softmax(scores.masked_fill(~mask, float("-inf")).float(), dim=-1)
    return torch.matmul(probabilities.to(query.dtype), value), mask


def heavily_compressed_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    block_size: int,
) -> Tensor:
    """HCA-style reference using causal block summaries and a current-prefix summary."""
    if query.shape != key.shape or key.shape != value.shape or query.ndim != 4:
        raise ValueError("query/key/value must share shape [B, H, T, D]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    _, _, sequence, head_dim = query.shape
    outputs = []
    for position in range(sequence):
        summaries_k = []
        summaries_v = []
        current_block = position // block_size
        for block_index in range(current_block):
            start = block_index * block_size
            end = start + block_size
            summaries_k.append(key[:, :, start:end].mean(dim=-2))
            summaries_v.append(value[:, :, start:end].mean(dim=-2))
        current_start = current_block * block_size
        summaries_k.append(key[:, :, current_start : position + 1].mean(dim=-2))
        summaries_v.append(value[:, :, current_start : position + 1].mean(dim=-2))
        compressed_key = torch.stack(summaries_k, dim=-2)
        compressed_value = torch.stack(summaries_v, dim=-2)
        scores = torch.einsum("bhd,bhkd->bhk", query[:, :, position], compressed_key) / math.sqrt(
            head_dim
        )
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        outputs.append(torch.einsum("bhk,bhkd->bhd", probabilities, compressed_value))
    return torch.stack(outputs, dim=-2)


class CompressedHybridAttentionLite(nn.Module):
    """DeepSeek-V4-inspired CSA/HCA teaching module with a learned output gate."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        index_dim: int,
        local_window: int,
        top_k: int,
        compression_block_size: int,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0 or index_dim <= 0:
            raise ValueError("invalid compressed hybrid attention dimensions")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.index_dim = index_dim
        self.local_window = local_window
        self.top_k = top_k
        self.compression_block_size = compression_block_size
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.index_q_proj = nn.Linear(d_model, n_heads * index_dim, bias=False)
        self.index_k_proj = nn.Linear(d_model, n_heads * index_dim, bias=False)
        self.branch_gate = nn.Linear(d_model, n_heads, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _heads(self, inputs: Tensor, dimension: int) -> Tensor:
        batch, sequence, _ = inputs.shape
        return inputs.view(batch, sequence, self.n_heads, dimension).transpose(1, 2)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Return hybrid output and the inspectable CSA token mask."""
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.d_model:
            raise ValueError("hidden_states must have shape [B, T, d_model]")
        query = self._heads(self.q_proj(hidden_states), self.head_dim)
        key = self._heads(self.k_proj(hidden_states), self.head_dim)
        value = self._heads(self.v_proj(hidden_states), self.head_dim)
        index_query = self._heads(self.index_q_proj(hidden_states), self.index_dim)
        index_key = self._heads(self.index_k_proj(hidden_states), self.index_dim)
        sparse, mask = compressed_sparse_attention(
            query,
            key,
            value,
            index_query,
            index_key,
            local_window=self.local_window,
            top_k=self.top_k,
        )
        compressed = heavily_compressed_attention(
            query, key, value, block_size=self.compression_block_size
        )
        gate = torch.sigmoid(self.branch_gate(hidden_states)).transpose(1, 2).unsqueeze(-1)
        mixed = gate * sparse + (1.0 - gate) * compressed
        batch, _, sequence, _ = mixed.shape
        merged = mixed.transpose(1, 2).contiguous().view(batch, sequence, self.d_model)
        return cast(Tensor, self.out_proj(merged)), mask


class HybridMixerStack(nn.Module):
    """Interleave several Gated DeltaNet layers with one full-attention layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        n_layers: int,
        linear_layers_per_attention: int = 3,
    ) -> None:
        super().__init__()
        if n_layers <= 0 or linear_layers_per_attention <= 0:
            raise ValueError("layer counts must be positive")
        self.d_model = d_model
        self.n_heads = n_heads
        self.layer_types: tuple[str, ...] = tuple(
            "attention" if (index + 1) % (linear_layers_per_attention + 1) == 0 else "delta"
            for index in range(n_layers)
        )
        modules: list[nn.Module] = []
        for layer_type in self.layer_types:
            if layer_type == "delta":
                modules.append(GatedDeltaNet(d_model, n_heads))
            else:
                modules.append(
                    nn.MultiheadAttention(d_model, n_heads, bias=False, batch_first=True)
                )
        self.layers = nn.ModuleList(modules)
        self.norms = nn.ModuleList(nn.RMSNorm(d_model) for _ in range(n_layers))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the configured 3:1-style hybrid schedule with residual updates."""
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.d_model:
            raise ValueError("hidden_states must have shape [B, T, d_model]")
        sequence = hidden_states.size(1)
        causal_mask = torch.ones(
            sequence, sequence, dtype=torch.bool, device=hidden_states.device
        ).triu(diagonal=1)
        for layer_type, layer, norm in zip(self.layer_types, self.layers, self.norms, strict=True):
            normalized = norm(hidden_states)
            if layer_type == "delta":
                assert isinstance(layer, GatedDeltaNet)
                update, _ = layer(normalized)
            else:
                assert isinstance(layer, nn.MultiheadAttention)
                update, _ = layer(
                    normalized,
                    normalized,
                    normalized,
                    attn_mask=causal_mask,
                    need_weights=False,
                )
            hidden_states = hidden_states + update
        return hidden_states
