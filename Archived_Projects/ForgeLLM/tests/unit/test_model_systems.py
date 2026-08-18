"""Tests for FlashAttention math, INT8 storage, and benchmark contracts."""

import torch
from torch.nn import functional as F

from forgellm.model.systems import (
    Int8WeightOnlyLinear,
    benchmark_callable,
    online_softmax_blockwise_attention,
    quantize_symmetric_int8,
)


def test_online_softmax_blockwise_attention_matches_sdpa_and_gradients() -> None:
    torch.manual_seed(40)
    query = torch.randn(2, 3, 11, 8, dtype=torch.float64, requires_grad=True)
    key = torch.randn(2, 3, 11, 8, dtype=torch.float64, requires_grad=True)
    value = torch.randn(2, 3, 11, 8, dtype=torch.float64, requires_grad=True)
    reference_query, reference_key, reference_value = (
        tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value)
    )

    actual = online_softmax_blockwise_attention(
        query, key, value, query_block_size=4, key_block_size=3
    )
    expected = F.scaled_dot_product_attention(
        reference_query, reference_key, reference_value, is_causal=True
    )
    actual.square().mean().backward()  # type: ignore[no-untyped-call]
    expected.square().mean().backward()  # type: ignore[no-untyped-call]

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    for tensor, reference in zip(
        (query, key, value),
        (reference_query, reference_key, reference_value),
        strict=True,
    ):
        torch.testing.assert_close(tensor.grad, reference.grad, rtol=1e-5, atol=1e-7)


def test_online_softmax_result_is_independent_of_tile_sizes() -> None:
    torch.manual_seed(41)
    query = torch.randn(1, 2, 9, 4)
    key = torch.randn(1, 2, 9, 4)
    value = torch.randn(1, 2, 9, 6)

    small_tiles = online_softmax_blockwise_attention(
        query, key, value, query_block_size=2, key_block_size=3
    )
    large_tiles = online_softmax_blockwise_attention(
        query, key, value, query_block_size=9, key_block_size=9
    )

    torch.testing.assert_close(small_tiles, large_tiles, rtol=1e-5, atol=1e-6)


def test_per_channel_int8_quantization_reduces_error_and_storage() -> None:
    torch.manual_seed(42)
    weight = torch.randn(12, 16) * torch.linspace(0.01, 10.0, 12)[:, None]
    per_tensor = quantize_symmetric_int8(weight, per_channel=False)
    per_channel = quantize_symmetric_int8(weight, per_channel=True)
    tensor_error = (per_tensor.dequantize() - weight).square().mean()
    channel_error = (per_channel.dequantize() - weight).square().mean()

    assert float(channel_error) < float(tensor_error)
    assert per_channel.values.shape == weight.shape
    assert per_channel.scale.shape == (12, 1)


def test_int8_weight_only_linear_approximates_fp32_and_uses_fewer_weight_bytes() -> None:
    torch.manual_seed(43)
    source = torch.nn.Linear(32, 24)
    quantized = Int8WeightOnlyLinear(source)
    inputs = torch.randn(4, 7, 32)

    expected = source(inputs)
    actual = quantized(inputs)

    assert float((actual - expected).abs().max()) < 0.02
    assert quantized.stored_weight_bytes < source.weight.numel() * source.weight.element_size()


def test_cpu_benchmark_records_positive_latency_without_cuda_memory() -> None:
    inputs = torch.randn(8, 8)
    result = benchmark_callable(torch.relu, inputs, warmup=1, repetitions=3)

    assert result.mean_milliseconds > 0
    assert result.peak_memory_mib is None
    assert result.repetitions == 3
