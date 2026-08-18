"""Autoregressive generation policies built on the ForgeLLM KV cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from forgellm.model.decoder import DecoderLM


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Sampling controls for educational autoregressive decoding."""

    max_new_tokens: int = 20
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    eos_token_id: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_new_tokens, bool) or self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative")


def filter_logits(logits: Tensor, *, top_k: int | None, top_p: float | None) -> Tensor:
    """Apply top-k and nucleus filters while always retaining at least one token."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, V]")
    filtered = logits.clone()
    if top_k is not None:
        keep = min(top_k, filtered.size(-1))
        threshold = torch.topk(filtered, keep, dim=-1).values[:, -1, None]
        filtered.masked_fill_(filtered < threshold, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, dim=-1, descending=True)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_logits.masked_fill_(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


def sample_next_token(
    logits: Tensor,
    config: GenerationConfig,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Select one token per batch row using greedy or stochastic sampling."""
    if config.temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    scaled = logits / config.temperature
    filtered = filter_logits(scaled, top_k=config.top_k, top_p=config.top_p)
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.no_grad()
def generate(model: DecoderLM, input_ids: Tensor, config: GenerationConfig) -> Tensor:
    """Generate tokens using one prefill followed by cached single-token decoding."""
    if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.size(1) == 0:
        raise ValueError("input_ids must be non-empty torch.long with shape [B, T]")
    if input_ids.size(1) + config.max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus generated tokens exceeds model max_seq_len")
    if config.eos_token_id is not None and config.eos_token_id >= model.config.vocab_size:
        raise ValueError("eos_token_id must be smaller than vocab_size")
    if config.max_new_tokens == 0:
        return input_ids.clone()

    generated = input_ids.clone()
    output = model(input_ids, use_cache=True)
    cache = output.cache
    if cache is None:
        raise RuntimeError("model did not return a cache during generation")

    generator = torch.Generator(device=input_ids.device)
    generator.manual_seed(config.seed)
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    for _ in range(config.max_new_tokens):
        next_token = sample_next_token(output.logits[:, -1, :], config, generator=generator)
        if config.eos_token_id is not None:
            eos_fill = torch.full_like(next_token, config.eos_token_id)
            next_token = torch.where(finished[:, None], eos_fill, next_token)
            finished |= next_token.squeeze(-1).eq(config.eos_token_id)
        generated = torch.cat((generated, next_token), dim=1)
        if bool(finished.all()):
            break
        output = model(next_token, cache=cache, use_cache=True)
        if output.cache is None:
            raise RuntimeError("model stopped returning a cache during generation")
        cache = output.cache
    return generated
