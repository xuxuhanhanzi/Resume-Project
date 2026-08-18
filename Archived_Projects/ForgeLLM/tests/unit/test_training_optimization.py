"""Optimizer partition and warmup/cosine scheduler tests."""

import torch
from torch import nn

from forgellm.training.config import TrainingConfig
from forgellm.training.optimization import (
    WarmupCosineScheduler,
    build_optimizer,
    partition_parameters,
)


class TinyParameters(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def test_adamw_partition_excludes_embedding_norm_and_bias_from_decay() -> None:
    module = TinyParameters()
    partition = partition_parameters(module.named_parameters(), use_muon=False)

    decay_names = {name for name, _ in partition.decay}
    no_decay_names = {name for name, _ in partition.no_decay}
    assert decay_names == {"linear.weight"}
    assert {"embedding.weight", "linear.bias", "norm.weight", "norm.bias"} == no_decay_names


def test_muon_owns_matrix_weights_while_adamw_keeps_vectors_and_embedding() -> None:
    module = TinyParameters()
    partition = partition_parameters(module.named_parameters(), use_muon=True)

    assert {name for name, _ in partition.matrix} == {"linear.weight"}
    assert "embedding.weight" in {name for name, _ in partition.no_decay}
    assert not partition.decay


def test_warmup_cosine_schedule_hits_documented_updates_and_restores() -> None:
    module = TinyParameters()
    config = TrainingConfig(learning_rate=1.0, warmup_steps=2, max_steps=6)
    optimizer = build_optimizer(module.named_parameters(), config)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=2,
        total_steps=6,
        min_lr_ratio=0.1,
    )

    assert scheduler.current_lrs() == [0.5, 0.5]
    scheduler.step()
    assert scheduler.current_lrs() == [1.0, 1.0]
    scheduler.step()
    for _ in range(4):
        scheduler.step()
    torch.testing.assert_close(torch.tensor(scheduler.current_lrs()), torch.tensor([0.1, 0.1]))

    restored_optimizer = build_optimizer(TinyParameters().named_parameters(), config)
    restored = WarmupCosineScheduler(
        restored_optimizer,
        warmup_steps=2,
        total_steps=6,
        min_lr_ratio=0.1,
    )
    restored.load_state_dict(scheduler.state_dict())
    assert restored.current_lrs() == scheduler.current_lrs()
