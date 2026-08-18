"""Small distributed-data-parallel correctness exercises for Stage 2."""

from __future__ import annotations

import os

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel


def average_gradient_tensors(replica_gradients: list[Tensor]) -> Tensor:
    """Return the arithmetic mean that DDP all-reduce should produce."""
    if not replica_gradients:
        raise ValueError("at least one replica gradient is required")
    shape = replica_gradients[0].shape
    if any(gradient.shape != shape for gradient in replica_gradients):
        raise ValueError("all replica gradients must have the same shape")
    return torch.stack(replica_gradients).mean(dim=0)


def _ddp_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["USE_LIBUV"] = "0"
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(50)
        model = nn.Linear(4, 2, bias=False)
        distributed_model = DistributedDataParallel(model)
        inputs = torch.arange(12, dtype=torch.float32).view(3, 4) + rank
        targets = torch.full((3, 2), float(rank))
        loss = torch.nn.functional.mse_loss(distributed_model(inputs), targets)
        loss.backward()  # type: ignore[no-untyped-call]
        gradient = model.weight.grad
        if gradient is None:
            raise RuntimeError("DDP did not create a weight gradient")
        gathered = [torch.zeros_like(gradient) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, gradient)
        if not all(torch.equal(gathered[0], other) for other in gathered[1:]):
            raise RuntimeError("DDP replicas received different averaged gradients")
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("DDP produced a non-finite gradient")
    finally:
        torch.distributed.destroy_process_group()


def run_cpu_ddp_smoke(*, world_size: int = 2, port: int = 29_531) -> None:
    """Launch a real multi-process Gloo DDP gradient-synchronization smoke test."""
    if world_size < 2 or not 1024 <= port <= 65535:
        raise ValueError("world_size must be at least two and port must be valid")
    torch.multiprocessing.spawn(  # type: ignore[attr-defined,no-untyped-call]
        _ddp_worker,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )
