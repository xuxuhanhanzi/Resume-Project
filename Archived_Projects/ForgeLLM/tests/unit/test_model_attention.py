"""Correctness tests for manual MHA/GQA, SDPA, causality, and KV caches."""

import pytest
import torch

from forgellm.model.attention import (
    CausalSelfAttention,
    causal_attention_mask,
    repeat_kv,
)
from forgellm.model.config import ModelConfig


def tiny_attention_config(*, backend: str = "manual", kv_heads: int = 2) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=kv_heads,
        ffn_hidden_dim=64,
        max_seq_len=16,
        attention_backend=backend,  # type: ignore[arg-type]
    )


def test_repeat_kv_maps_adjacent_query_heads_to_the_same_kv_head() -> None:
    inputs = torch.tensor([[[[1.0]], [[2.0]]]])

    repeated = repeat_kv(inputs, repeats=2)

    assert repeated[:, :, 0, 0].tolist() == [[1.0, 1.0, 2.0, 2.0]]


def test_cache_aware_causal_mask_exposes_all_past_but_no_future() -> None:
    mask = causal_attention_mask(2, 5, past_length=3, device=torch.device("cpu"))

    assert mask.tolist() == [
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]


def test_manual_attention_matches_pytorch_sdpa() -> None:
    torch.manual_seed(2)
    manual = CausalSelfAttention(tiny_attention_config(backend="manual"))
    sdpa = CausalSelfAttention(tiny_attention_config(backend="sdpa"))
    sdpa.load_state_dict(manual.state_dict())
    hidden = torch.randn(2, 7, 32, requires_grad=True)
    hidden_reference = hidden.detach().clone().requires_grad_(True)

    manual_output, _ = manual(hidden)
    sdpa_output, _ = sdpa(hidden_reference)
    manual_output.square().mean().backward()
    sdpa_output.square().mean().backward()

    torch.testing.assert_close(manual_output, sdpa_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(hidden.grad, hidden_reference.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_attention_is_causal(backend: str) -> None:
    torch.manual_seed(3)
    module = CausalSelfAttention(tiny_attention_config(backend=backend))
    prefix = torch.randn(1, 4, 32)
    extended = torch.cat((prefix, torch.randn(1, 3, 32)), dim=1)

    prefix_output, _ = module(prefix)
    extended_output, _ = module(extended)

    torch.testing.assert_close(prefix_output, extended_output[:, :4], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_incremental_kv_cache_matches_full_attention(backend: str) -> None:
    torch.manual_seed(4)
    module = CausalSelfAttention(tiny_attention_config(backend=backend)).eval()
    hidden = torch.randn(2, 8, 32)

    full_output, _ = module(hidden)
    cached_steps = []
    cache = None
    for position in range(hidden.size(1)):
        step_output, cache = module(hidden[:, position : position + 1], cache=cache, use_cache=True)
        cached_steps.append(step_output)

    assert cache is not None
    assert cache.key.shape == (2, 2, 8, 8)
    torch.testing.assert_close(torch.cat(cached_steps, dim=1), full_output, rtol=2e-5, atol=2e-6)


def test_mha_configuration_uses_one_kv_head_per_query_head() -> None:
    module = CausalSelfAttention(tiny_attention_config(kv_heads=4))
    hidden = torch.randn(2, 5, 32)

    output, cache = module(hidden, use_cache=True)

    assert output.shape == hidden.shape
    assert cache is not None
    assert cache.key.shape == (2, 4, 5, 8)


def test_qk_norm_path_preserves_attention_contract_and_gradients() -> None:
    config = ModelConfig(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=64,
        max_seq_len=16,
        qk_norm=True,
    )
    module = CausalSelfAttention(config)
    hidden = torch.randn(2, 5, 32, requires_grad=True)

    output, _ = module(hidden)
    output.square().mean().backward()

    assert output.shape == hidden.shape
    assert hidden.grad is not None and bool(torch.isfinite(hidden.grad).all())
