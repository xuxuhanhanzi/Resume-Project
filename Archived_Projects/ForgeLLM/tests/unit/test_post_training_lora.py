"""Dependency-free LoRA correctness, Artifact and merge tests."""

from pathlib import Path

import pytest
import torch
from torch import nn

from forgellm.post_training.lora import (
    LoRAError,
    LoRALinear,
    adapter_state_dict,
    inject_lora,
    load_adapter,
    save_adapter,
)


def test_lora_initializes_as_exact_noop_and_freezes_base() -> None:
    torch.manual_seed(4)
    base = nn.Linear(6, 5, bias=False, dtype=torch.float64)
    inputs = torch.randn(3, 6, dtype=torch.float64)
    expected = base(inputs).detach().clone()

    adapter = LoRALinear(base, rank=2, alpha=4.0)

    assert torch.equal(adapter(inputs), expected)
    assert not adapter.base_layer.weight.requires_grad
    assert adapter.lora_a.requires_grad and adapter.lora_b.requires_grad
    assert torch.count_nonzero(adapter.lora_b).item() == 0


def test_lora_gradient_updates_adapter_without_base_gradient() -> None:
    base = nn.Linear(4, 3, bias=False)
    adapter = LoRALinear(base, rank=2, alpha=2.0)
    inputs = torch.randn(5, 4)

    adapter(inputs).square().mean().backward()

    assert adapter.base_layer.weight.grad is None
    assert adapter.lora_b.grad is not None
    assert torch.isfinite(adapter.lora_b.grad).all()


def test_merge_and_unmerge_preserve_outputs_with_registered_tolerance() -> None:
    torch.manual_seed(7)
    adapter = LoRALinear(nn.Linear(4, 3, bias=False, dtype=torch.float64), rank=2, alpha=2.0)
    nn.init.normal_(adapter.lora_b)
    inputs = torch.randn(2, 4, dtype=torch.float64)
    expected = adapter(inputs).detach().clone()
    base_before = adapter.base_layer.weight.detach().clone()

    adapter.merge()
    merged = adapter(inputs)
    adapter.unmerge()
    restored = adapter(inputs)

    assert torch.allclose(merged, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(restored, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(adapter.base_layer.weight, base_before, atol=1e-12, rtol=0.0)


def test_injection_counts_only_low_rank_parameters() -> None:
    model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))

    injection = inject_lora(model, rank=2, alpha=4.0, exclude_suffixes=())

    assert injection.module_names == ("0", "2")
    assert injection.trainable_parameters == 2 * (8 + 6) + 2 * (6 + 4)
    assert injection.trainable_parameters == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert all(not module.base_layer.weight.requires_grad for module in injection.modules)


def test_adapter_artifact_round_trip(tmp_path: Path) -> None:
    first = nn.Sequential(nn.Linear(4, 4, bias=False))
    second = nn.Sequential(nn.Linear(4, 4, bias=False))
    first.load_state_dict(second.state_dict())
    inject_lora(first, rank=2, alpha=2.0, exclude_suffixes=())
    inject_lora(second, rank=2, alpha=2.0, exclude_suffixes=())
    first_adapter = first[0]
    assert isinstance(first_adapter, LoRALinear)
    nn.init.normal_(first_adapter.lora_b)
    path = tmp_path / "adapter.pt"

    save_adapter(path, first, metadata={"run": "unit"})
    metadata = load_adapter(path, second)

    assert metadata == {"run": "unit"}
    for key, value in adapter_state_dict(first).items():
        assert torch.equal(adapter_state_dict(second)[key], value)


def test_invalid_rank_and_double_merge_fail_fast() -> None:
    with pytest.raises(LoRAError):
        LoRALinear(nn.Linear(2, 2), rank=3, alpha=1.0)
    adapter = LoRALinear(nn.Linear(2, 2), rank=1, alpha=1.0)
    adapter.merge()
    with pytest.raises(LoRAError, match="already merged"):
        adapter.merge()
