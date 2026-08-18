"""Opt-in tests for the compiled C++/CUDA silu_mul operator."""

import os

import pytest
import torch
from torch.nn import functional as F

from forgellm.custom_ops.silu_mul import load_silu_mul_extension

pytestmark = pytest.mark.skipif(
    os.environ.get("FORGELLM_TEST_EXTENSION") != "1",
    reason="set FORGELLM_TEST_EXTENSION=1 after activating the compiler toolchain",
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compiled_silu_mul_forward_backward_and_opcheck(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    load_silu_mul_extension(with_cuda=torch.cuda.is_available())
    gate = torch.randn(4, 7, device=device, dtype=torch.float64, requires_grad=True)
    value = torch.randn(4, 7, device=device, dtype=torch.float64, requires_grad=True)

    actual = torch.ops.forgellm_ops.silu_mul(gate, value)
    expected = F.silu(gate) * value

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.autograd.gradcheck(torch.ops.forgellm_ops.silu_mul, (gate, value))
    torch.library.opcheck(torch.ops.forgellm_ops.silu_mul, (gate.detach(), value.detach()))


def test_compiled_silu_mul_accepts_empty_tensors() -> None:
    load_silu_mul_extension(with_cuda=torch.cuda.is_available())
    for device in (torch.device("cpu"), torch.device("cuda")):
        if device.type == "cuda" and not torch.cuda.is_available():
            continue
        gate = torch.empty(0, 3, device=device)
        value = torch.empty_like(gate)
        actual = torch.ops.forgellm_ops.silu_mul(gate, value)
        assert actual.shape == (0, 3)
        assert actual.device.type == device.type
