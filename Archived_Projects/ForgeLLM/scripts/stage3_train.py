"""Run a bounded Stage 3 pretraining experiment and emit auditable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch

from forgellm.model.generation import GenerationConfig, generate
from forgellm.tokenization.bpe import EOS_ID, TokenizerError
from forgellm.training.config import load_experiment_config
from forgellm.training.factory import create_trainer
from forgellm.training.trainer import summarize_history


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize_complete_log(path: Path) -> dict[str, object]:
    def number(record: dict[str, object], key: str) -> float:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"training log field {key!r} must be numeric")
        return float(value)

    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: object = json.loads(line)
        if isinstance(raw, dict) and raw.get("event") == "train_step":
            records.append(raw)
    if not records:
        return {"steps": 0}
    losses = [number(record, "loss") for record in records]
    throughputs = [number(record, "tokens_per_second") for record in records]
    peaks = [
        0.0 if record.get("peak_memory_mib") is None else number(record, "peak_memory_mib")
        for record in records
    ]
    first = losses[: min(10, len(losses))]
    last = losses[-min(10, len(losses)) :]
    steps = [int(number(record, "step")) for record in records]
    return {
        "steps": len(records),
        "first_window_mean_loss": sum(first) / len(first),
        "last_window_mean_loss": sum(last) / len(last),
        "loss_declined": sum(last) / len(last) < sum(first) / len(first),
        "mean_tokens_per_second": sum(throughputs) / len(throughputs),
        "peak_memory_mib": max(peaks),
        "all_losses_finite": all(math.isfinite(loss) for loss in losses),
        "logged_steps_contiguous": steps == list(range(1, len(steps) + 1)),
    }


def _fixed_generations(trainer: object) -> list[dict[str, object]]:
    from forgellm.training.trainer import Trainer

    if not isinstance(trainer, Trainer):
        raise TypeError("trainer must be a Trainer")
    prompts = ("Once upon a time", "The little dog")
    records: list[dict[str, object]] = []
    trainer.model.eval()
    for prompt in prompts:
        prompt_ids = trainer.tokenizer.encode(prompt, add_bos=True)
        available = trainer.model.config.max_seq_len - len(prompt_ids)
        if available <= 0:
            records.append({"prompt": prompt, "error": "prompt exceeds model context"})
            continue
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=trainer.device)
        generated = (
            generate(
                trainer.model,
                input_ids,
                GenerationConfig(
                    max_new_tokens=min(32, available),
                    temperature=0.0,
                    eos_token_id=EOS_ID,
                ),
            )[0]
            .cpu()
            .tolist()
        )
        try:
            decoded = trainer.tokenizer.decode(generated)
            records.append({"prompt": prompt, "text": decoded, "token_ids": generated})
        except TokenizerError as error:
            records.append({"prompt": prompt, "error": str(error), "token_ids": generated})
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args(argv)

    experiment = load_experiment_config(args.config)
    checkpoint = args.output_dir / "latest.pt"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError("output already has a checkpoint; pass --resume or choose a new run")
    trainer = create_trainer(
        experiment,
        tokenizer_path=args.tokenizer,
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output_dir,
    )
    if args.resume:
        trainer.load_checkpoint(checkpoint)
    history, stop_reason = trainer.train(stop_after_step=args.stop_after_step)
    final_validation = trainer.validate()
    report = {
        "schema_version": "forgellm-stage3-run-report-v1",
        "claim_boundary": "Small-scale pretraining-loop evidence only; not a model-quality claim.",
        "config": experiment.as_dict(),
        "config_fingerprint": experiment.fingerprint(),
        "data": {
            "train_path": str(args.train),
            "train_sha256": _sha256(args.train),
            "train_dataset_fingerprint": trainer.train_stream.dataset.fingerprint,
            "validation_path": str(args.validation),
            "validation_sha256": _sha256(args.validation),
            "validation_dataset_fingerprint": trainer.validation_dataset.fingerprint,
        },
        "final_state": trainer.state.state_dict(),
        "final_validation": final_validation.as_dict(),
        "fixed_generations": _fixed_generations(trainer),
        "stop_reason": stop_reason,
        "session_summary": summarize_history(history),
        "summary": _summarize_complete_log(trainer.log_path),
        "tokenizer_fingerprint": trainer.tokenizer.fingerprint(),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report_path.resolve())
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
