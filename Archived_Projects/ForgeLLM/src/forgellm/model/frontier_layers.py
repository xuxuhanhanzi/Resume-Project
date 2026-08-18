"""Reference layers for MoE, modern residual paths, and multi-token prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from forgellm.model.layers import SwiGLU


@dataclass(frozen=True, slots=True)
class MoEOutput:
    """Sparse expert output plus inspectable routing evidence."""

    hidden_states: Tensor
    router_logits: Tensor
    selected_experts: Tensor
    expert_counts: Tensor
    auxiliary_loss: Tensor


def hash_expert_indices(token_ids: Tensor, num_experts: int) -> Tensor:
    """Map token IDs deterministically to experts for Hash-MoE bootstrapping."""
    if token_ids.ndim != 2 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be torch.long with shape [B, T]")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    return torch.remainder(token_ids * 2_654_435_761 + 1_013_904_223, num_experts)


class SparseMoE(nn.Module):
    """Top-k routed SwiGLU experts with an optional always-on shared expert."""

    def __init__(
        self,
        d_model: int,
        expert_hidden_dim: int,
        *,
        num_experts: int,
        top_k: int,
        shared_expert: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or expert_hidden_dim <= 0 or num_experts <= 0:
            raise ValueError("MoE dimensions and num_experts must be positive")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d_model, expert_hidden_dim) for _ in range(num_experts))
        self.shared_expert = SwiGLU(d_model, expert_hidden_dim) if shared_expert else None

    def forward(
        self,
        hidden_states: Tensor,
        *,
        token_ids: Tensor | None = None,
        use_hash_routing: bool = False,
    ) -> MoEOutput:
        """Route flattened tokens and combine only their selected expert outputs."""
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.d_model:
            raise ValueError("hidden_states must have shape [B, T, d_model]")
        batch, sequence, _ = hidden_states.shape
        flattened = hidden_states.reshape(-1, self.d_model)
        router_logits = self.router(flattened)
        router_probabilities = torch.softmax(router_logits.float(), dim=-1)

        if use_hash_routing:
            if self.top_k != 1:
                raise ValueError("hash routing reference currently requires top_k=1")
            if token_ids is None or token_ids.shape != (batch, sequence):
                raise ValueError("hash routing requires token_ids matching [B, T]")
            selected = hash_expert_indices(token_ids, self.num_experts).reshape(-1, 1)
            weights = torch.ones_like(selected, dtype=hidden_states.dtype)
        else:
            weights, selected = torch.topk(router_probabilities, self.top_k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            weights = weights.to(dtype=hidden_states.dtype)

        combined = torch.zeros_like(flattened)
        for expert_index, expert in enumerate(self.experts):
            token_indices, slots = torch.where(selected.eq(expert_index))
            if token_indices.numel() == 0:
                continue
            expert_output = expert(flattened[token_indices])
            weighted = expert_output * weights[token_indices, slots, None]
            combined.index_add_(0, token_indices, weighted)
        if self.shared_expert is not None:
            combined = combined + self.shared_expert(flattened)

        counts = torch.bincount(selected.flatten(), minlength=self.num_experts)
        assignment_fraction = counts.float() / selected.numel()
        mean_router_probability = router_probabilities.mean(dim=0)
        auxiliary_loss = self.num_experts * torch.sum(assignment_fraction * mean_router_probability)
        return MoEOutput(
            hidden_states=combined.view(batch, sequence, self.d_model),
            router_logits=router_logits.view(batch, sequence, self.num_experts),
            selected_experts=selected.view(batch, sequence, self.top_k),
            expert_counts=counts,
            auxiliary_loss=auxiliary_loss,
        )


def sinkhorn_doubly_stochastic(logits: Tensor, *, iterations: int = 8) -> Tensor:
    """Project positive mixing weights toward row/column-normalized matrices."""
    if logits.ndim != 2 or logits.size(0) != logits.size(1):
        raise ValueError("logits must be a square matrix")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    matrix = torch.exp(logits - logits.max())
    for _ in range(iterations):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True)
    return matrix


class ManifoldHyperConnectionLite(nn.Module):
    """Small mHC-inspired residual stream mixer with a constrained matrix.

    This teaches the central invariant—non-negative, approximately doubly
    stochastic stream mixing—without claiming exact reproduction of the
    distributed DeepSeek-V4 implementation.
    """

    def __init__(self, stream_count: int, d_model: int) -> None:
        super().__init__()
        if stream_count <= 0 or d_model <= 0:
            raise ValueError("stream_count and d_model must be positive")
        self.stream_count = stream_count
        self.d_model = d_model
        initial = torch.eye(stream_count) * 4.0
        self.mixing_logits = nn.Parameter(initial)
        self.branch_logits = nn.Parameter(torch.zeros(stream_count))

    def mixing_matrix(self) -> Tensor:
        """Return the constrained residual-stream mixing matrix."""
        return sinkhorn_doubly_stochastic(self.mixing_logits)

    def forward(self, streams: Tensor, branch_output: Tensor) -> Tensor:
        """Mix old streams and distribute a transformed branch among them."""
        if streams.ndim != 4 or streams.shape[-2:] != (
            self.stream_count,
            self.d_model,
        ):
            raise ValueError("streams must have shape [B, T, stream_count, d_model]")
        if branch_output.shape != streams.shape[:2] + (self.d_model,):
            raise ValueError("branch_output must have shape [B, T, d_model]")
        mixed = torch.einsum("ij,btjd->btid", self.mixing_matrix(), streams)
        branch_weights = torch.softmax(self.branch_logits, dim=0)
        return mixed + branch_output.unsqueeze(-2) * branch_weights[None, None, :, None]


class AttentionResiduals(nn.Module):
    """Choose a content-dependent mixture of previous layer residual states."""

    def __init__(self, d_model: int, attention_dim: int) -> None:
        super().__init__()
        if d_model <= 0 or attention_dim <= 0:
            raise ValueError("d_model and attention_dim must be positive")
        self.d_model = d_model
        self.attention_dim = attention_dim
        self.query_proj = nn.Linear(d_model, attention_dim, bias=False)
        self.key_proj = nn.Linear(d_model, attention_dim, bias=False)

    def forward(self, query_state: Tensor, residual_history: Tensor) -> tuple[Tensor, Tensor]:
        """Return the weighted residual history and its layer-selection weights."""
        if query_state.ndim != 3 or query_state.size(-1) != self.d_model:
            raise ValueError("query_state must have shape [B, T, d_model]")
        if residual_history.ndim != 4 or residual_history.shape[:2] != query_state.shape[:2]:
            raise ValueError("residual_history must have shape [B, T, L, d_model]")
        if residual_history.size(-1) != self.d_model or residual_history.size(-2) == 0:
            raise ValueError("residual_history must contain at least one compatible layer")
        query = self.query_proj(query_state).unsqueeze(-2)
        keys = self.key_proj(residual_history)
        scores = (query * keys).sum(dim=-1) / math.sqrt(self.attention_dim)
        weights = torch.softmax(scores.float(), dim=-1).to(query_state.dtype)
        selected = torch.sum(residual_history * weights.unsqueeze(-1), dim=-2)
        return selected, weights


class MultiTokenPredictionHead(nn.Module):
    """Predict several future token offsets from every current hidden state."""

    def __init__(self, d_model: int, vocab_size: int, num_future_tokens: int) -> None:
        super().__init__()
        if d_model <= 0 or vocab_size <= 0 or num_future_tokens <= 0:
            raise ValueError("MTP dimensions must be positive")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_future_tokens = num_future_tokens
        self.projection = nn.Linear(d_model, num_future_tokens * vocab_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Return logits with shape ``[B, T, future_offset, vocab]``."""
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.d_model:
            raise ValueError("hidden_states must have shape [B, T, d_model]")
        batch, sequence, _ = hidden_states.shape
        projected = self.projection(hidden_states)
        return cast(
            Tensor,
            projected.view(batch, sequence, self.num_future_tokens, self.vocab_size),
        )


def multi_token_prediction_loss(logits: Tensor, input_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Average valid cross entropy for offsets ``+1`` through ``+K``."""
    if logits.ndim != 4 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("logits must be [B, T, K, V] aligned with input_ids [B, T]")
    _, sequence, future_count, vocab_size = logits.shape
    losses = []
    for offset in range(1, future_count + 1):
        if sequence <= offset:
            raise ValueError("sequence must be longer than every predicted offset")
        prediction = logits[:, :-offset, offset - 1, :].contiguous()
        target = input_ids[:, offset:].contiguous()
        losses.append(F.cross_entropy(prediction.view(-1, vocab_size), target.view(-1)))
    per_offset = torch.stack(losses)
    return per_offset.mean(), per_offset
