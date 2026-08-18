"""Mathematical building blocks used by the educational decoder."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization without mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, inputs: Tensor) -> Tensor:
        """Normalize the last dimension and preserve the input dtype."""
        compute_dtype = (
            torch.float32 if inputs.dtype in (torch.float16, torch.bfloat16) else inputs.dtype
        )
        promoted = inputs.to(dtype=compute_dtype)
        variance = promoted.pow(2).mean(dim=-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.to(dtype=compute_dtype)).to(dtype=inputs.dtype)


def rotate_half(inputs: Tensor) -> Tensor:
    """Rotate each adjacent pair ``(x0, x1)`` into ``(-x1, x0)``."""
    if inputs.size(-1) % 2 != 0:
        raise ValueError("RoPE requires an even last dimension")
    paired = inputs.reshape(*inputs.shape[:-1], inputs.size(-1) // 2, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) frequency provider and applicator."""

    inv_freq: Tensor

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if theta <= 0:
            raise ValueError("theta must be positive")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", frequencies, persistent=False)

    def angles(self, positions: Tensor, *, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Return cosine and sine tables shaped ``[1, 1, T, head_dim]``."""
        if positions.ndim != 1:
            raise ValueError("positions must be a rank-1 tensor")
        angles = torch.outer(positions.float(), self.inv_freq.float())
        repeated = torch.repeat_interleave(angles, repeats=2, dim=-1)
        return (
            repeated.cos()[None, None, :, :].to(dtype=dtype),
            repeated.sin()[None, None, :, :].to(dtype=dtype),
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply the same position-dependent rotation to query and key."""
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("query and key must have shape [B, H, T, head_dim]")
        if query.shape[-2:] != key.shape[-2:]:
            raise ValueError("query and key must share sequence and head dimensions")
        if query.size(-1) != self.head_dim:
            raise ValueError("query/key last dimension does not match RoPE head_dim")
        if positions.numel() != query.size(-2):
            raise ValueError("positions length must match the query sequence length")
        cos, sin = self.angles(positions.to(device=query.device), dtype=query.dtype)
        return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, d_model: int, hidden_dim: int, *, bias: bool = False) -> None:
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the gated feed-forward transformation."""
        return cast(
            Tensor,
            self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs)),
        )
