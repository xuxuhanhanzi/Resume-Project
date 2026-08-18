"""Dependency-free LoRA Linear, injection and adapter Artifact helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRAError(ValueError):
    """Raised when adapter configuration or state is inconsistent."""


class LoRALinear(nn.Module):
    """Frozen Linear plus ``scaling * B(A(dropout(x)))``."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0 or rank > min(base_layer.in_features, base_layer.out_features):
            raise LoRAError("rank must be positive and no larger than the base dimensions")
        if alpha <= 0:
            raise LoRAError("alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise LoRAError("dropout must be in [0, 1)")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(
            torch.empty(
                rank,
                base_layer.in_features,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                rank,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )
        self.merged = False
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def delta_weight(self) -> Tensor:
        """Return the uncast low-rank update in parameter dtype."""
        return (self.lora_b @ self.lora_a) * self.scaling

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply base and adapter unless the adapter is already merged."""
        result = self.base_layer(inputs)
        if self.merged:
            return cast(Tensor, result)
        adapted = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return cast(Tensor, result + adapted * self.scaling)

    @torch.no_grad()
    def merge(self) -> None:
        """Add the adapter update to the frozen base exactly once."""
        if self.merged:
            raise LoRAError("adapter is already merged")
        self.base_layer.weight.add_(self.delta_weight().to(self.base_layer.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Remove a previously merged adapter update."""
        if not self.merged:
            raise LoRAError("adapter is not merged")
        self.base_layer.weight.sub_(self.delta_weight().to(self.base_layer.weight.dtype))
        self.merged = False


@dataclass(frozen=True, slots=True)
class LoRAInjection:
    """Names and modules replaced by one injection operation."""

    module_names: tuple[str, ...]
    modules: tuple[LoRALinear, ...]
    trainable_parameters: int


def _resolve_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        child = parent._modules.get(part)
        if child is None:
            raise LoRAError(f"could not resolve module path {qualified_name!r}")
        parent = child
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_suffixes: Sequence[str] | None = None,
    exclude_suffixes: Sequence[str] = ("lm_head",),
) -> LoRAInjection:
    """Replace selected Linear modules while freezing every original parameter."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if any(name.endswith(suffix) for suffix in exclude_suffixes):
            continue
        if target_suffixes is not None and not any(
            name.endswith(suffix) for suffix in target_suffixes
        ):
            continue
        candidates.append((name, module))
    if not candidates:
        raise LoRAError("no Linear modules matched the LoRA target policy")
    names: list[str] = []
    modules: list[LoRALinear] = []
    for name, base_layer in candidates:
        parent, child_name = _resolve_parent(model, name)
        adapter = LoRALinear(base_layer, rank=rank, alpha=alpha, dropout=dropout)
        parent._modules[child_name] = adapter
        names.append(name)
        modules.append(adapter)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected = sum(
        module.rank * (module.base_layer.in_features + module.base_layer.out_features)
        for module in modules
    )
    if trainable != expected:
        raise LoRAError(f"trainable parameter count {trainable} does not match {expected}")
    return LoRAInjection(tuple(names), tuple(modules), trainable)


def iter_lora_modules(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    """Yield injected adapters in stable module order."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def adapter_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return CPU clones of only the LoRA matrices."""
    state: dict[str, Tensor] = {}
    for name, module in iter_lora_modules(model):
        state[f"{name}.lora_a"] = module.lora_a.detach().cpu().clone()
        state[f"{name}.lora_b"] = module.lora_b.detach().cpu().clone()
    if not state:
        raise LoRAError("model has no LoRA modules")
    return state


def load_adapter_state_dict(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Strictly restore adapter matrices without modifying Base weights."""
    expected = adapter_state_dict(model)
    if set(state) != set(expected):
        raise LoRAError("adapter state keys do not match injected modules")
    modules = dict(iter_lora_modules(model))
    with torch.no_grad():
        for key, tensor in state.items():
            module_name, parameter_name = key.rsplit(".", maxsplit=1)
            module = modules[module_name]
            parameter = module.lora_a if parameter_name == "lora_a" else module.lora_b
            if tensor.shape != parameter.shape:
                raise LoRAError(f"adapter tensor shape mismatch for {key}")
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def save_adapter(path: Path, model: nn.Module, *, metadata: Mapping[str, object]) -> None:
    """Save a small schema-tagged Adapter Artifact without overwriting evidence."""
    if path.exists():
        raise FileExistsError(f"adapter Artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "forgellm-lora-adapter-v1",
        "metadata": dict(metadata),
        "state": adapter_state_dict(model),
    }
    torch.save(payload, path)


def load_adapter(path: Path, model: nn.Module) -> dict[str, object]:
    """Load and validate a Stage 4 Adapter Artifact."""
    raw: object = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or set(raw) != {"metadata", "schema_version", "state"}:
        raise LoRAError("adapter Artifact has unexpected fields")
    payload = cast(dict[str, object], raw)
    if payload["schema_version"] != "forgellm-lora-adapter-v1":
        raise LoRAError("unsupported adapter Artifact schema")
    state = payload["state"]
    metadata = payload["metadata"]
    if not isinstance(state, dict) or not isinstance(metadata, dict):
        raise LoRAError("adapter Artifact state/metadata must be mappings")
    if not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()):
        raise LoRAError("adapter Artifact contains invalid tensors")
    load_adapter_state_dict(model, cast(dict[str, Tensor], state))
    return cast(dict[str, object], metadata)
