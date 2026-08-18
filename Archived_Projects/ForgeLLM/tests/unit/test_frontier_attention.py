"""Tests for modern attention reference laboratories."""

import math

import torch

from forgellm.model.attention import manual_scaled_dot_product_attention
from forgellm.model.frontier_attention import (
    CompressedHybridAttentionLite,
    GatedDeltaNet,
    HybridMixerStack,
    MultiHeadLatentAttention,
    ScaledRotaryEmbedding,
    moba_reference_attention,
)


def test_linear_rope_scaling_maps_doubled_position_to_original_angle() -> None:
    original = ScaledRotaryEmbedding(8, original_max_seq_len=8, max_seq_len=16, mode="none")
    scaled = ScaledRotaryEmbedding(
        8,
        original_max_seq_len=8,
        max_seq_len=16,
        scaling_factor=2.0,
        mode="linear",
    )
    inputs = torch.randn(1, 2, 1, 8)

    original_position_one = original(inputs, torch.tensor([1]))
    scaled_position_two = scaled(inputs, torch.tensor([2]))

    torch.testing.assert_close(scaled_position_two, original_position_one)


def test_dynamic_ntk_matches_original_inside_trained_context() -> None:
    baseline = ScaledRotaryEmbedding(8, original_max_seq_len=8, max_seq_len=16, mode="none")
    dynamic = ScaledRotaryEmbedding(
        8,
        original_max_seq_len=8,
        max_seq_len=16,
        scaling_factor=2.0,
        mode="dynamic_ntk",
    )
    inputs = torch.randn(1, 2, 8, 8)
    positions = torch.arange(8)

    torch.testing.assert_close(dynamic(inputs, positions), baseline(inputs, positions))


def test_mla_compressed_cache_matches_full_sequence_forward() -> None:
    torch.manual_seed(20)
    module = MultiHeadLatentAttention(
        d_model=32,
        n_heads=4,
        q_rank=12,
        kv_rank=8,
        nope_dim=6,
        rope_dim=2,
        value_dim=8,
        max_seq_len=16,
    ).eval()
    hidden = torch.randn(2, 7, 32)

    full, _ = module(hidden)
    pieces = []
    cache = None
    for position in range(hidden.size(1)):
        output, cache = module(hidden[:, position : position + 1], cache=cache, use_cache=True)
        pieces.append(output)

    assert cache is not None
    assert module.compressed_cache_elements_per_token == 10
    assert module.mha_cache_elements_per_token == 64
    assert cache.latent_kv.shape == (2, 7, 8)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=2e-5, atol=2e-6)


def test_moba_becomes_full_causal_attention_when_all_blocks_are_selected() -> None:
    torch.manual_seed(21)
    query = torch.randn(1, 2, 8, 4)
    key = torch.randn(1, 2, 8, 4)
    value = torch.randn(1, 2, 8, 4)

    moba_output, mask = moba_reference_attention(query, key, value, block_size=2, top_k_blocks=4)
    causal = torch.ones(8, 8, dtype=torch.bool).tril()
    full_output = manual_scaled_dot_product_attention(query, key, value, allowed_mask=causal)

    torch.testing.assert_close(moba_output, full_output)
    assert torch.equal(mask[0, 0], causal)


def test_sparse_moba_mask_is_causal_and_reduces_visible_keys() -> None:
    torch.manual_seed(22)
    query = torch.randn(1, 1, 12, 4)
    key = torch.randn(1, 1, 12, 4)
    value = torch.randn(1, 1, 12, 4)

    output, mask = moba_reference_attention(query, key, value, block_size=3, top_k_blocks=2)

    assert output.shape == query.shape
    assert not bool(mask.triu(diagonal=1).any())
    assert int(mask.sum()) < int(torch.ones(12, 12, dtype=torch.bool).tril().sum())


def test_delta_net_token_recurrence_matches_full_call() -> None:
    torch.manual_seed(23)
    module = GatedDeltaNet(d_model=16, n_heads=4).eval()
    hidden = torch.randn(2, 9, 16)

    full, full_state = module(hidden)
    state = None
    pieces = []
    for position in range(hidden.size(1)):
        output, state = module(hidden[:, position : position + 1], state=state)
        pieces.append(output)

    assert state is not None
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state, full_state, rtol=1e-5, atol=1e-6)


def test_delta_net_is_causal() -> None:
    torch.manual_seed(24)
    module = GatedDeltaNet(d_model=16, n_heads=4).eval()
    prefix = torch.randn(1, 5, 16)
    extended = torch.cat((prefix, torch.randn(1, 4, 16)), dim=1)

    prefix_output, _ = module(prefix)
    extended_output, _ = module(extended)

    torch.testing.assert_close(prefix_output, extended_output[:, :5])
    assert math.isfinite(float(extended_output.square().mean()))


def test_csa_hca_lite_is_causal_and_exposes_sparse_mask() -> None:
    torch.manual_seed(25)
    module = CompressedHybridAttentionLite(
        d_model=16,
        n_heads=4,
        index_dim=3,
        local_window=3,
        top_k=2,
        compression_block_size=3,
    ).eval()
    prefix = torch.randn(1, 7, 16)
    extended = torch.cat((prefix, torch.randn(1, 4, 16)), dim=1)

    prefix_output, prefix_mask = module(prefix)
    extended_output, extended_mask = module(extended)

    torch.testing.assert_close(prefix_output, extended_output[:, :7], rtol=1e-5, atol=1e-6)
    assert not bool(prefix_mask.triu(diagonal=1).any())
    assert extended_mask.shape == (1, 4, 11, 11)


def test_hybrid_stack_uses_three_delta_layers_then_full_attention() -> None:
    torch.manual_seed(26)
    module = HybridMixerStack(
        d_model=16,
        n_heads=4,
        n_layers=8,
        linear_layers_per_attention=3,
    ).eval()
    prefix = torch.randn(1, 5, 16)
    extended = torch.cat((prefix, torch.randn(1, 3, 16)), dim=1)

    prefix_output = module(prefix)
    extended_output = module(extended)

    assert module.layer_types == (
        "delta",
        "delta",
        "delta",
        "attention",
        "delta",
        "delta",
        "delta",
        "attention",
    )
    torch.testing.assert_close(prefix_output, extended_output[:, :5], rtol=2e-5, atol=2e-6)
