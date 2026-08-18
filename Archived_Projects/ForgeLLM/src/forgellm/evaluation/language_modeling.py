"""Tokenizer-aware language-modeling totals and comparison gates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class LanguageModelingTotals:
    """Additive NLL totals with both token and UTF-8 byte denominators."""

    negative_log_likelihood: float
    target_tokens: int
    target_bytes: int
    tokenizer_sha256: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.negative_log_likelihood) or self.negative_log_likelihood < 0:
            raise ValueError("negative log-likelihood must be finite and non-negative")
        if self.target_tokens <= 0 or self.target_bytes <= 0:
            raise ValueError("language-modeling denominators must be positive")
        if len(self.tokenizer_sha256) != 64:
            raise ValueError("tokenizer_sha256 must be complete")

    @property
    def loss_per_token(self) -> float:
        """Return token-normalized NLL."""
        return self.negative_log_likelihood / self.target_tokens

    @property
    def perplexity(self) -> float:
        """Return token perplexity, meaningful only within one tokenizer identity."""
        return math.exp(min(self.loss_per_token, 80.0))

    @property
    def bits_per_byte(self) -> float:
        """Return UTF-8 byte-normalized NLL in bits."""
        return self.negative_log_likelihood / (self.target_bytes * math.log(2.0))

    def as_dict(self) -> dict[str, float | int | str]:
        """Return totals and all derived metrics."""
        return {
            **asdict(self),
            "bits_per_byte": self.bits_per_byte,
            "loss_per_token": self.loss_per_token,
            "perplexity": self.perplexity,
        }


def merge_language_modeling_totals(
    totals: list[LanguageModelingTotals],
) -> LanguageModelingTotals:
    """Aggregate by additive numerators/denominators, never by averaging PPL."""
    if not totals:
        raise ValueError("at least one language-modeling total is required")
    tokenizer = totals[0].tokenizer_sha256
    if any(item.tokenizer_sha256 != tokenizer for item in totals):
        raise ValueError("cannot merge language-modeling totals from different tokenizers")
    return LanguageModelingTotals(
        negative_log_likelihood=sum(item.negative_log_likelihood for item in totals),
        target_tokens=sum(item.target_tokens for item in totals),
        target_bytes=sum(item.target_bytes for item in totals),
        tokenizer_sha256=tokenizer,
    )


def assert_perplexity_comparable(
    left: LanguageModelingTotals, right: LanguageModelingTotals
) -> None:
    """Fail before directly ranking PPL values from different tokenizers."""
    if left.tokenizer_sha256 != right.tokenizer_sha256:
        raise RuntimeError(
            "perplexity comparison is invalid across tokenizer identities; compare BPB instead"
        )
