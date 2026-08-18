"""Always-on tests for the Python contract behind the custom operator."""

import pytest
import torch
from torch.nn import functional as F

from forgellm.custom_ops.silu_mul import silu_mul_reference


def test_silu_mul_reference_matches_explicit_equation_and_gradient() -> None:
    gate = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    value = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    expected_gate = gate.detach().clone().requires_grad_(True)
    expected_value = value.detach().clone().requires_grad_(True)

    actual = silu_mul_reference(gate, value)
    expected = F.silu(expected_gate) * expected_value
    actual.sum().backward()  # type: ignore[no-untyped-call]
    expected.sum().backward()  # type: ignore[no-untyped-call]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(gate.grad, expected_gate.grad)
    torch.testing.assert_close(value.grad, expected_value.grad)


def test_silu_mul_reference_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        silu_mul_reference(torch.ones(2), torch.ones(3))
