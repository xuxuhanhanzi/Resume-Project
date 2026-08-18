"""Systems-oriented reference code for attention, quantization, and benchmarking."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def online_softmax_blockwise_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_block_size: int,
    key_block_size: int,
    causal: bool = True,
) -> Tensor:
    """Exact tiled attention using the online-softmax recurrence.

    This exposes the mathematical core behind FlashAttention: retain only a
    running row maximum, normalization denominator, and value accumulator for
    each query tile. It does not claim kernel-level speedups in Python.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must have shape [B, H, T, D]")
    if key.shape[:-1] != value.shape[:-1] or query.shape[:2] != key.shape[:2]:
        raise ValueError("query/key/value batch and head dimensions must align")
    if query.size(-1) != key.size(-1):
        raise ValueError("query and key head dimensions must match")
    if query_block_size <= 0 or key_block_size <= 0:
        raise ValueError("block sizes must be positive")
    if causal and query.size(-2) != key.size(-2):
        raise ValueError("causal reference currently requires equal query/key lengths")

    _, _, query_length, _ = query.shape
    key_length = key.size(-2)
    value_dim = value.size(-1)
    scale = 1.0 / math.sqrt(query.size(-1))
    compute_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    output_blocks = []
    for query_start in range(0, query_length, query_block_size):
        query_end = min(query_length, query_start + query_block_size)
        query_block = query[:, :, query_start:query_end].to(dtype=compute_dtype)
        block_rows = query_end - query_start
        running_max = torch.full(
            query.shape[:2] + (block_rows, 1),
            float("-inf"),
            dtype=compute_dtype,
            device=query.device,
        )
        running_sum = torch.zeros_like(running_max)
        accumulator = torch.zeros(
            query.shape[:2] + (block_rows, value_dim),
            dtype=compute_dtype,
            device=query.device,
        )
        key_limit = query_end if causal else key_length
        for key_start in range(0, key_limit, key_block_size):
            key_end = min(key_length, key_start + key_block_size)
            key_block = key[:, :, key_start:key_end].to(dtype=compute_dtype)
            value_block = value[:, :, key_start:key_end].to(dtype=compute_dtype)
            scores = torch.matmul(query_block, key_block.transpose(-2, -1)) * scale
            if causal:
                query_positions = torch.arange(
                    query_start, query_end, device=query.device
                ).unsqueeze(-1)
                key_positions = torch.arange(key_start, key_end, device=query.device).unsqueeze(0)
                scores = scores.masked_fill(key_positions > query_positions, float("-inf"))

            tile_max = scores.max(dim=-1, keepdim=True).values
            next_max = torch.maximum(running_max, tile_max)
            old_scale = torch.exp(running_max - next_max)
            tile_probabilities = torch.exp(scores - next_max)
            next_sum = old_scale * running_sum + tile_probabilities.sum(dim=-1, keepdim=True)
            accumulator = old_scale * accumulator + torch.matmul(tile_probabilities, value_block)
            running_max = next_max
            running_sum = next_sum
        output_blocks.append(accumulator / running_sum)
    return torch.cat(output_blocks, dim=-2).to(dtype=query.dtype)


@dataclass(frozen=True, slots=True)
class QuantizedWeight:
    """Symmetric signed INT8 weight and its dequantization scale."""

    values: Tensor
    scale: Tensor
    per_channel: bool

    def __post_init__(self) -> None:
        if self.values.dtype != torch.int8 or not self.scale.is_floating_point():
            raise ValueError("values must be int8 and scale must be floating point")

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> Tensor:
        """Reconstruct a floating-point approximation of the weight."""
        return self.values.to(dtype=dtype) * self.scale.to(dtype=dtype)


def quantize_symmetric_int8(weight: Tensor, *, per_channel: bool) -> QuantizedWeight:
    """Quantize a rank-2 weight with symmetric per-tensor or output-channel scales."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point matrix")
    reduction_dim: int | tuple[int, ...] = 1 if per_channel else (0, 1)
    maximum = weight.detach().abs().amax(dim=reduction_dim, keepdim=True)
    scale = torch.clamp(maximum / 127.0, min=torch.finfo(torch.float32).eps)
    values = torch.round(weight.detach() / scale).clamp(-127, 127).to(torch.int8)
    return QuantizedWeight(values=values, scale=scale.float(), per_channel=per_channel)


class Int8WeightOnlyLinear(nn.Module):
    """Inspectable weight-only INT8 Linear that dequantizes during forward.

    It demonstrates storage and error behavior. Because dequantization occurs in
    PyTorch at every call, this reference is not an optimized deployment kernel.
    """

    qweight: Tensor
    scale: Tensor

    def __init__(self, source: nn.Linear, *, per_channel: bool = True) -> None:
        super().__init__()
        quantized = quantize_symmetric_int8(source.weight, per_channel=per_channel)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_buffer("qweight", quantized.values)
        self.register_buffer("scale", quantized.scale)
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Dequantize to the input dtype and apply a normal linear operation."""
        weight = self.qweight.to(dtype=inputs.dtype) * self.scale.to(dtype=inputs.dtype)
        return F.linear(inputs, weight, self.bias)

    @property
    def stored_weight_bytes(self) -> int:
        """Bytes occupied by quantized values and scales, excluding bias."""
        return (
            self.qweight.numel() * self.qweight.element_size()
            + self.scale.numel() * self.scale.element_size()
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Minimal reproducible latency and CUDA-memory measurement."""

    mean_milliseconds: float
    peak_memory_mib: float | None
    repetitions: int


def benchmark_callable(
    function: object,
    *args: Tensor,
    warmup: int = 5,
    repetitions: int = 20,
) -> BenchmarkResult:
    """Benchmark a callable with fixed inputs and required CUDA synchronization."""
    if not callable(function):
        raise TypeError("function must be callable")
    if warmup < 0 or repetitions <= 0:
        raise ValueError("warmup must be non-negative and repetitions positive")
    uses_cuda = any(argument.is_cuda for argument in args)
    for _ in range(warmup):
        function(*args)
    if uses_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repetitions):
        function(*args)
    if uses_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if uses_cuda else None
    return BenchmarkResult(
        mean_milliseconds=elapsed * 1000.0 / repetitions,
        peak_memory_mib=peak_memory,
        repetitions=repetitions,
    )
