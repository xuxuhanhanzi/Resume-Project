"""Stage 3 training-loop, accumulation, validation, MTP, and resume gates."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from forgellm.model.config import ModelConfig
from forgellm.tokenization.bpe import ByteBPETokenizer
from forgellm.tokenization.config import TokenizerConfig
from forgellm.training.config import ExperimentConfig, TrainingConfig
from forgellm.training.factory import create_trainer
from forgellm.training.trainer import Trainer, TrainingDivergedError


def _write_jsonl(path: Path, prefix: str, documents: int = 12) -> None:
    lines = [
        json.dumps(
            {
                "id": f"{prefix}-{index}",
                "text": (f"{prefix} story {index}: the small cat ran home safely. " * 5),
            },
            sort_keys=True,
        )
        for index in range(documents)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture(
    tmp_path: Path, training: TrainingConfig | None = None
) -> tuple[ExperimentConfig, Path, Path, Path]:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_jsonl(train_path, "train")
    _write_jsonl(validation_path, "validation", documents=4)
    texts = [train_path.read_text(encoding="utf-8"), validation_path.read_text(encoding="utf-8")]
    tokenizer = ByteBPETokenizer.train(TokenizerConfig(280, 2), texts)
    tokenizer.save(tokenizer_path)
    model = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=64,
        max_seq_len=16,
        attention_backend="sdpa",
    )
    resolved_training = training or TrainingConfig(
        run_name="integration",
        seed=71,
        device="cpu",
        precision="fp32",
        sequence_length=16,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=4,
        learning_rate=0.003,
        warmup_steps=1,
        validation_interval=4,
        validation_batches=2,
        checkpoint_interval=0,
    )
    return ExperimentConfig(model, resolved_training), tokenizer_path, train_path, validation_path


def _trainer(
    experiment: ExperimentConfig,
    tokenizer: Path,
    train: Path,
    validation: Path,
    output: Path,
) -> Trainer:
    return create_trainer(
        experiment,
        tokenizer_path=tokenizer,
        train_path=train,
        validation_path=validation,
        output_dir=output,
    )


def test_validation_changes_neither_parameters_nor_training_cursor(tmp_path: Path) -> None:
    experiment, tokenizer, train, validation = _fixture(tmp_path)
    trainer = _trainer(experiment, tokenizer, train, validation, tmp_path / "run")
    trainer.train_step()
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
    }
    cursor_before = trainer.train_stream.state_dict()

    metrics = trainer.validate()

    assert metrics.loss > 0 and metrics.perplexity > 1
    assert metrics.bits_per_byte is not None and metrics.bits_per_byte > 0
    assert cursor_before == trainer.train_stream.state_dict()
    assert trainer.model.training
    assert all(
        torch.equal(parameters_before[name], parameter)
        for name, parameter in trainer.model.named_parameters()
    )


def test_gradient_accumulation_matches_one_large_batch_update(tmp_path: Path) -> None:
    base_training = TrainingConfig(
        run_name="accumulation",
        seed=72,
        device="cpu",
        precision="fp32",
        sequence_length=16,
        micro_batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=1,
        learning_rate=0.001,
        warmup_steps=0,
        validation_interval=2,
        validation_batches=1,
        checkpoint_interval=0,
    )
    experiment, tokenizer, train, validation = _fixture(tmp_path, base_training)
    large = _trainer(experiment, tokenizer, train, validation, tmp_path / "large")
    accumulated_experiment = ExperimentConfig(
        experiment.model,
        replace(
            base_training,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        ),
    )
    accumulated = _trainer(
        accumulated_experiment, tokenizer, train, validation, tmp_path / "accumulated"
    )

    large.train_step()
    accumulated.train_step()

    for (large_name, large_parameter), (accum_name, accum_parameter) in zip(
        large.model.named_parameters(), accumulated.model.named_parameters(), strict=True
    ):
        assert large_name == accum_name
        torch.testing.assert_close(large_parameter, accum_parameter, atol=1e-7, rtol=1e-6)


def test_cpu_fp32_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    experiment, tokenizer, train, validation = _fixture(tmp_path)
    continuous = _trainer(experiment, tokenizer, train, validation, tmp_path / "continuous")
    continuous_losses = [continuous.train_step().loss for _ in range(4)]

    interrupted = _trainer(experiment, tokenizer, train, validation, tmp_path / "interrupted")
    resumed_losses = [interrupted.train_step().loss for _ in range(2)]
    checkpoint = interrupted.save_checkpoint()
    resumed = _trainer(experiment, tokenizer, train, validation, tmp_path / "resumed")
    resumed.load_checkpoint(checkpoint)
    resumed_losses.extend(resumed.train_step().loss for _ in range(2))

    assert continuous_losses == resumed_losses
    for expected, actual in zip(
        continuous.model.parameters(), resumed.model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)
    assert torch.equal(
        continuous.train_stream.next_batch().token_ids,
        resumed.train_stream.next_batch().token_ids,
    )


def test_mtp_auxiliary_head_participates_in_real_optimizer_step(tmp_path: Path) -> None:
    training = TrainingConfig(
        run_name="mtp",
        seed=73,
        device="cpu",
        precision="fp32",
        sequence_length=16,
        micro_batch_size=2,
        max_steps=1,
        learning_rate=0.001,
        warmup_steps=0,
        validation_interval=2,
        validation_batches=1,
        checkpoint_interval=0,
        mtp_future_tokens=2,
        mtp_loss_weight=0.1,
    )
    experiment, tokenizer, train, validation = _fixture(tmp_path, training)
    trainer = _trainer(experiment, tokenizer, train, validation, tmp_path / "mtp")
    assert trainer.mtp_head is not None
    before = trainer.mtp_head.projection.weight.detach().clone()

    metrics = trainer.train_step()

    assert metrics.mtp_loss is not None and metrics.mtp_loss > 0
    assert not torch.equal(before, trainer.mtp_head.projection.weight)


def test_non_finite_loss_fails_fast_before_optimizer_step(tmp_path: Path) -> None:
    experiment, tokenizer, train, validation = _fixture(tmp_path)
    trainer = _trainer(experiment, tokenizer, train, validation, tmp_path / "nan")
    with torch.no_grad():
        next(trainer.model.parameters()).fill_(float("nan"))

    with pytest.raises(TrainingDivergedError, match="non-finite loss"):
        trainer.train_step()
    assert trainer.state.step == 0


def test_manual_step_limit_saves_the_exact_partial_state(tmp_path: Path) -> None:
    experiment, tokenizer, train, validation = _fixture(tmp_path)
    trainer = _trainer(experiment, tokenizer, train, validation, tmp_path / "partial")

    history, reason = trainer.train(stop_after_step=2)

    assert reason == "manual_step_limit"
    assert len(history) == 2
    assert trainer.state.step == 2
    assert trainer.checkpoint_path.is_file()
