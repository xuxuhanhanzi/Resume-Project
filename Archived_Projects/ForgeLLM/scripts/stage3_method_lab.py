"""Run short, single-variable AdamW/Muon and single-token/MTP comparisons."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from forgellm.training.config import ExperimentConfig, OptimizerName, load_experiment_config
from forgellm.training.factory import create_trainer
from forgellm.training.trainer import summarize_history


def _run_variant(
    base: ExperimentConfig,
    *,
    name: str,
    seed: int,
    optimizer: OptimizerName,
    mtp_future_tokens: int,
    mtp_loss_weight: float,
    tokenizer_path: Path,
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    steps: int,
) -> dict[str, object]:
    training = replace(
        base.training,
        run_name=f"stage3-method-{name}-seed-{seed}",
        seed=seed,
        device="auto",
        precision="fp32",
        optimizer=optimizer,
        max_steps=steps,
        max_tokens=None,
        max_duration_seconds=None,
        warmup_steps=min(10, steps),
        validation_interval=steps,
        checkpoint_interval=0,
        mtp_future_tokens=mtp_future_tokens,
        mtp_loss_weight=mtp_loss_weight,
    )
    experiment = ExperimentConfig(base.model, training)
    trainer = create_trainer(
        experiment,
        tokenizer_path=tokenizer_path,
        train_path=train_path,
        validation_path=validation_path,
        output_dir=output_dir / name / f"seed_{seed}",
    )
    history, stop_reason = trainer.train()
    validation = trainer.validate()
    mtp_parameters = (
        sum(parameter.numel() for parameter in trainer.mtp_head.parameters())
        if trainer.mtp_head is not None
        else 0
    )
    return {
        "name": name,
        "seed": seed,
        "optimizer": optimizer,
        "mtp_future_tokens": mtp_future_tokens,
        "mtp_loss_weight": mtp_loss_weight,
        "base_model_parameters": trainer.model.parameter_count(),
        "mtp_parameters": mtp_parameters,
        "stop_reason": stop_reason,
        "summary": summarize_history(history),
        "validation": validation.as_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    args = parser.parse_args(argv)
    if args.steps <= 0 or not args.seeds:
        raise ValueError("steps and seed collection must be non-empty and positive")
    if (args.output_dir / "report.json").exists():
        raise FileExistsError("method-lab report already exists; choose a new output directory")

    base = load_experiment_config(args.config)
    variants: tuple[tuple[str, OptimizerName, int, float], ...] = (
        ("adamw_single", "adamw", 0, 0.0),
        ("muon_single", "muon", 0, 0.0),
        ("adamw_mtp", "adamw", 2, 0.1),
    )
    runs = [
        _run_variant(
            base,
            name=name,
            seed=seed,
            optimizer=optimizer,
            mtp_future_tokens=future_tokens,
            mtp_loss_weight=loss_weight,
            tokenizer_path=args.tokenizer,
            train_path=args.train,
            validation_path=args.validation,
            output_dir=args.output_dir,
            steps=args.steps,
        )
        for name, optimizer, future_tokens, loss_weight in variants
        for seed in args.seeds
    ]
    aggregate: dict[str, object] = {}
    for name, _, _, _ in variants:
        selected = [run for run in runs if run["name"] == name]
        validation_losses = [
            float(run["validation"]["loss"])  # type: ignore[index]
            for run in selected
        ]
        aggregate[name] = {
            "mean_validation_loss": statistics.fmean(validation_losses),
            "stdev_validation_loss": (
                statistics.stdev(validation_losses) if len(validation_losses) > 1 else 0.0
            ),
            "seeds": list(args.seeds),
        }
    report = {
        "schema_version": "forgellm-stage3-method-lab-v1",
        "claim_boundary": (
            "Short teaching runs test implementation and stability. A lower mean is not by "
            "itself evidence of general superiority."
        ),
        "fixed_settings": {
            "config": str(args.config),
            "steps": args.steps,
            "seeds": list(args.seeds),
        },
        "primary_variables": {
            "optimizer_lab": "AdamW versus Muon; MTP disabled",
            "objective_lab": "single-token versus two-offset MTP; AdamW fixed",
        },
        "runs": runs,
        "aggregate": aggregate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report_path.resolve())
    print(json.dumps(aggregate, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
