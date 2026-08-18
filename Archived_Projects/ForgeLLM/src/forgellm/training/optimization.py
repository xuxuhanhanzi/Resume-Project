"""Parameter grouping, hybrid Muon/AdamW, and an inspectable LR schedule."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.optim import Optimizer

from forgellm.model.optim import Muon
from forgellm.training.config import TrainingConfig


class OptimizationError(ValueError):
    """Raised when optimizer parameter ownership or state is inconsistent."""


@dataclass(frozen=True, slots=True)
class ParameterPartition:
    """Disjoint named parameter collections used by Stage 3 optimizers."""

    matrix: tuple[tuple[str, nn.Parameter], ...]
    decay: tuple[tuple[str, nn.Parameter], ...]
    no_decay: tuple[tuple[str, nn.Parameter], ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(
            name for group in (self.matrix, self.decay, self.no_decay) for name, _ in group
        )


def partition_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]], *, use_muon: bool
) -> ParameterPartition:
    """Assign each unique trainable tensor to exactly one documented group."""
    matrix: list[tuple[str, nn.Parameter]] = []
    decay: list[tuple[str, nn.Parameter]] = []
    no_decay: list[tuple[str, nn.Parameter]] = []
    seen: set[int] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        normalized = name.lower()
        excluded_from_decay = (
            parameter.ndim < 2
            or "norm" in normalized
            or "embedding" in normalized
            or normalized.endswith("bias")
        )
        eligible_for_muon = use_muon and parameter.ndim >= 2 and "embedding" not in normalized
        if eligible_for_muon:
            matrix.append((name, parameter))
        elif excluded_from_decay:
            no_decay.append((name, parameter))
        else:
            decay.append((name, parameter))
    if not seen:
        raise OptimizationError("no trainable parameters were supplied")
    partition = ParameterPartition(tuple(matrix), tuple(decay), tuple(no_decay))
    if len(set(partition.names)) != len(partition.names):
        raise OptimizationError("parameter names must be unique")
    return partition


class OptimizerBundle:
    """One logical optimizer that may contain AdamW plus a Muon instance."""

    def __init__(
        self, optimizers: list[tuple[str, Optimizer]], partition: ParameterPartition
    ) -> None:
        if not optimizers:
            raise OptimizationError("optimizer bundle must be non-empty")
        self.optimizers = tuple(optimizers)
        self.partition = partition

    def __iter__(self) -> Iterator[Optimizer]:
        return (optimizer for _, optimizer in self.optimizers)

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for _, optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for _, optimizer in self.optimizers:
            optimizer.step()

    @property
    def param_groups(self) -> list[dict[str, object]]:
        return [group for _, optimizer in self.optimizers for group in optimizer.param_groups]

    def state_dict(self) -> dict[str, object]:
        return {
            "names": [name for name, _ in self.optimizers],
            "states": [optimizer.state_dict() for _, optimizer in self.optimizers],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        names = state.get("names")
        states = state.get("states")
        expected_names = [name for name, _ in self.optimizers]
        if (
            names != expected_names
            or not isinstance(states, list)
            or len(states) != len(self.optimizers)
        ):
            raise OptimizationError("optimizer checkpoint layout does not match")
        for (_, optimizer), optimizer_state in zip(self.optimizers, states, strict=True):
            if not isinstance(optimizer_state, dict):
                raise OptimizationError("optimizer checkpoint state must be a mapping")
            optimizer.load_state_dict(optimizer_state)


def build_optimizer(
    named_parameters: Iterable[tuple[str, nn.Parameter]], config: TrainingConfig
) -> OptimizerBundle:
    """Build AdamW or the documented Muon-matrix/AdamW-fallback hybrid."""
    partition = partition_parameters(named_parameters, use_muon=config.optimizer == "muon")
    optimizers: list[tuple[str, Optimizer]] = []
    adam_groups: list[dict[str, object]] = []
    if partition.decay:
        adam_groups.append(
            {
                "params": [parameter for _, parameter in partition.decay],
                "weight_decay": config.weight_decay,
                "group_name": "adamw_decay",
            }
        )
    if partition.no_decay:
        adam_groups.append(
            {
                "params": [parameter for _, parameter in partition.no_decay],
                "weight_decay": 0.0,
                "group_name": "adamw_no_decay",
            }
        )
    if adam_groups:
        optimizers.append(
            (
                "adamw",
                torch.optim.AdamW(
                    adam_groups,
                    lr=config.learning_rate,
                    betas=(config.beta1, config.beta2),
                ),
            )
        )
    if partition.matrix:
        optimizers.append(
            (
                "muon",
                Muon(
                    [parameter for _, parameter in partition.matrix],
                    lr=config.muon_learning_rate,
                    momentum=0.95,
                    weight_decay=config.weight_decay,
                ),
            )
        )
    return OptimizerBundle(optimizers, partition)


class WarmupCosineScheduler:
    """Per-update linear warmup followed by cosine decay to a fixed ratio."""

    def __init__(
        self,
        optimizer: OptimizerBundle,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float,
    ) -> None:
        if warmup_steps < 0 or total_steps <= 0 or warmup_steps > total_steps:
            raise OptimizationError("scheduler step counts are invalid")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise OptimizationError("min_lr_ratio must be in [0, 1]")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.completed_steps = 0
        self.base_lrs = [float(cast(float, group["lr"])) for group in optimizer.param_groups]
        self._apply_for_next_update()

    def scale_for_update(self, update_number: int) -> float:
        """Return the multiplier for one-indexed optimizer update ``update_number``."""
        if update_number <= 0:
            raise OptimizationError("update_number must be positive")
        if self.warmup_steps and update_number <= self.warmup_steps:
            return update_number / self.warmup_steps
        if self.total_steps == self.warmup_steps:
            return self.min_lr_ratio
        progress = (update_number - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply_for_next_update(self) -> None:
        scale = self.scale_for_update(self.completed_steps + 1)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * scale

    def step(self) -> None:
        self.completed_steps += 1
        self._apply_for_next_update()

    def current_lrs(self) -> list[float]:
        return [float(cast(float, group["lr"])) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, object]:
        return {
            "completed_steps": self.completed_steps,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        fixed = (state.get("warmup_steps"), state.get("total_steps"), state.get("min_lr_ratio"))
        expected = (self.warmup_steps, self.total_steps, self.min_lr_ratio)
        if fixed != expected or state.get("base_lrs") != self.base_lrs:
            raise OptimizationError("scheduler checkpoint settings do not match")
        completed_steps = state.get("completed_steps")
        if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
            raise OptimizationError("scheduler completed_steps is invalid")
        if not 0 <= completed_steps <= self.total_steps:
            raise OptimizationError("scheduler completed_steps is out of range")
        self.completed_steps = completed_steps
        self._apply_for_next_update()
