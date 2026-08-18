"""Formula-level tests for RMSNorm, RoPE, and SwiGLU."""

import torch
from torch.nn import functional as F

from forgellm.model.layers import RMSNorm, RotaryEmbedding, SwiGLU


def test_rms_norm_matches_explicit_formula_and_preserves_dtype() -> None:
    inputs = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]], dtype=torch.float32)
    module = RMSNorm(4, eps=1e-5)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([1.0, 0.5, 2.0, -1.0]))

    expected = inputs * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + 1e-5)
    expected = expected * module.weight
    actual = module(inputs)

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == inputs.dtype


def test_rms_norm_passes_double_precision_gradcheck() -> None:
    module = RMSNorm(4).double()
    inputs = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(module, (inputs,))


def test_rope_position_zero_is_identity_and_rotation_preserves_norm() -> None:
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    query = torch.randn(2, 4, 3, 8)
    key = torch.randn(2, 2, 3, 8)
    positions = torch.arange(3)

    rotated_query, rotated_key = rope(query, key, positions)

    torch.testing.assert_close(rotated_query[:, :, 0], query[:, :, 0])
    torch.testing.assert_close(rotated_key[:, :, 0], key[:, :, 0])
    torch.testing.assert_close(
        rotated_query.square().sum(dim=-1), query.square().sum(dim=-1), rtol=1e-5, atol=1e-6
    )


def test_swiglu_matches_three_explicit_linear_operations() -> None:
    torch.manual_seed(1)
    module = SwiGLU(d_model=4, hidden_dim=6)
    inputs = torch.randn(2, 3, 4)

    expected = F.linear(
        F.silu(F.linear(inputs, module.gate_proj.weight)) * F.linear(inputs, module.up_proj.weight),
        module.down_proj.weight,
    )

    torch.testing.assert_close(module(inputs), expected)
