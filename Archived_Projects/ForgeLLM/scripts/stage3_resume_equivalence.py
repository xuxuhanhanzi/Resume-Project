"""Prove uninterrupted and save/reload CPU-FP32 training are exactly equivalent."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import torch

from forgellm.training.config import ExperimentConfig, load_experiment_config
from forgellm.training.factory import create_trainer


def _equal_nested(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_equal_nested(left[key], right[key]) for key in left)
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        return len(left) == len(right) and all(
            _equal_nested(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _model_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return set(left_state) == set(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--split-step", type=int, default=3)
    args = parser.parse_args(argv)
    if not 0 < args.split_step < args.steps:
        raise ValueError("require 0 < split-step < steps")

    loaded = load_experiment_config(args.config)
    training = replace(
        loaded.training,
        device="cpu",
        precision="fp32",
        max_steps=args.steps,
        max_tokens=None,
        max_duration_seconds=None,
        validation_interval=args.steps + 1,
        checkpoint_interval=0,
    )
    experiment = ExperimentConfig(model=loaded.model, training=training)

    continuous = create_trainer(
        experiment,
        tokenizer_path=args.tokenizer,
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output_dir / "continuous",
    )
    continuous_losses = [continuous.train_step().loss for _ in range(args.steps)]

    interrupted = create_trainer(
        experiment,
        tokenizer_path=args.tokenizer,
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output_dir / "interrupted",
    )
    resumed_losses = [interrupted.train_step().loss for _ in range(args.split_step)]
    checkpoint_path = interrupted.save_checkpoint()
    resumed = create_trainer(
        experiment,
        tokenizer_path=args.tokenizer,
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output_dir / "resumed",
    )
    resumed.load_checkpoint(checkpoint_path)
    resumed_losses.extend(resumed.train_step().loss for _ in range(args.steps - args.split_step))

    continuous_next = continuous.train_stream.next_batch()
    resumed_next = resumed.train_stream.next_batch()
    checks = {
        "losses_exact": continuous_losses == resumed_losses,
        "model_exact": _model_equal(continuous.model, resumed.model),
        "optimizer_exact": _equal_nested(
            continuous.optimizer.state_dict(), resumed.optimizer.state_dict()
        ),
        "scheduler_exact": _equal_nested(
            continuous.scheduler.state_dict(), resumed.scheduler.state_dict()
        ),
        "trainer_state_exact": continuous.state.state_dict() == resumed.state.state_dict(),
        "next_batch_indices_exact": torch.equal(
            continuous_next.sample_indices, resumed_next.sample_indices
        ),
        "next_batch_tokens_exact": torch.equal(continuous_next.token_ids, resumed_next.token_ids),
    }
    report = {
        "schema_version": "forgellm-stage3-resume-equivalence-v1",
        "steps": args.steps,
        "split_step": args.split_step,
        "checks": checks,
        "passed": all(checks.values()),
        "continuous_losses": continuous_losses,
        "resumed_losses": resumed_losses,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report_path.resolve())
    print(json.dumps(checks, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
