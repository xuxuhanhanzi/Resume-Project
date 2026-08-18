"""Validated configuration for the ForgeLLM educational decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

AttentionBackend = Literal["manual", "sdpa"]


class ModelConfigError(ValueError):
    """Raised when model dimensions or options are inconsistent."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """All architecture choices needed to construct a Decoder-only LM.

    The defaults deliberately describe a sub-million-parameter teaching model.
    Stage 3 can reuse the same implementation with a larger validated config.
    """

    vocab_size: int = 320
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    ffn_hidden_dim: int = 384
    max_seq_len: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    qk_norm: bool = False
    attention_backend: AttentionBackend = "manual"

    def __post_init__(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "ffn_hidden_dim": self.ffn_hidden_dim,
            "max_seq_len": self.max_seq_len,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ModelConfigError(f"{name} must be an integer")
            if value <= 0:
                raise ModelConfigError(f"{name} must be positive")

        if self.d_model % self.n_heads != 0:
            raise ModelConfigError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ModelConfigError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ModelConfigError("attention head_dim must be even for RoPE")
        if not isinstance(self.rms_norm_eps, (int, float)) or self.rms_norm_eps <= 0:
            raise ModelConfigError("rms_norm_eps must be positive")
        if not isinstance(self.rope_theta, (int, float)) or self.rope_theta <= 0:
            raise ModelConfigError("rope_theta must be positive")
        if not isinstance(self.dropout, (int, float)) or not 0.0 <= self.dropout < 1.0:
            raise ModelConfigError("dropout must be in [0, 1)")
        if not isinstance(self.tie_embeddings, bool) or not isinstance(self.qk_norm, bool):
            raise ModelConfigError("tie_embeddings and qk_norm must be booleans")
        if self.attention_backend not in ("manual", "sdpa"):
            raise ModelConfigError("attention_backend must be 'manual' or 'sdpa'")

    @property
    def head_dim(self) -> int:
        """Dimension of one attention head."""
        return self.d_model // self.n_heads

    @property
    def queries_per_kv(self) -> int:
        """Number of query heads sharing each key/value head."""
        return self.n_heads // self.n_kv_heads

    def as_dict(self) -> dict[str, int | float | bool | str]:
        """Return a JSON-compatible resolved configuration."""
        return asdict(self)
