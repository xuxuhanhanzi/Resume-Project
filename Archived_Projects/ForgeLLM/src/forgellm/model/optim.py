"""Educational Muon optimizer components for controlled Stage 2 experiments."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, cast

import torch
from torch import Tensor
from torch.optim import Optimizer


def zeropower_via_newton_schulz5(matrix: Tensor, *, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the orthogonal polar factor of a matrix with Newton–Schulz steps."""
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization requires a matrix")
    if steps <= 0 or eps <= 0:
        raise ValueError("steps and eps must be positive")
    original_dtype = matrix.dtype
    work = matrix.float()
    transposed = work.size(0) > work.size(1)
    if transposed:
        work = work.T
    work = work / (work.norm() + eps)
    coefficient_a, coefficient_b, coefficient_c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = work @ work.T
        polynomial = coefficient_b * gram + coefficient_c * (gram @ gram)
        work = coefficient_a * work + polynomial @ work
    if transposed:
        work = work.T
    return cast(Tensor, work.to(dtype=original_dtype))


class Muon(Optimizer):
    """Minimal Muon optimizer for teaching matrix update orthogonalization.

    Matrix parameters use a momentum update followed by Newton–Schulz
    orthogonalization. Vector/scalar parameters use the momentum update without
    orthogonalization. Production training normally applies a separate AdamW
    group to embeddings, norms, and biases.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        if lr <= 0 or not 0.0 <= momentum < 1.0 or weight_decay < 0 or ns_steps <= 0:
            raise ValueError("invalid Muon hyperparameters")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:  # type: ignore[override]
        """Perform one optimization step and optionally return closure loss."""
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state: dict[str, Any] = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer: Tensor = state["momentum_buffer"]
                buffer.mul_(momentum).add_(gradient, alpha=1.0 - momentum)
                update = gradient.lerp(buffer, momentum) if nesterov else buffer

                if update.ndim >= 2:
                    matrix = update.reshape(update.size(0), -1)
                    matrix = zeropower_via_newton_schulz5(matrix, steps=ns_steps)
                    aspect_scale = math.sqrt(max(1.0, matrix.size(0) / matrix.size(1)))
                    update = matrix.reshape_as(update) * aspect_scale
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(update, alpha=-lr)
        return loss
