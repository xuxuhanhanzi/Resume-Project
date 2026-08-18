"""Run one immutable bounded Qwen LoRA or QLoRA Stage 4 experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from forgellm.post_training.adapters import load_stage4_hf_stack
from forgellm.post_training.config import load_stage4_run_config
from forgellm.post_training.hf_experiment import (
    assert_stack_ready,
    build_run_provenance,
    evaluate_stage4,
    tokenize_records,
    train_bounded,
)
from forgellm.post_training.schema import load_instruction_jsonl


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate Base, train adapters, evaluate again and persist all evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report_path = args.output_dir / "report.json"
    if args.output_dir.exists():
        raise FileExistsError(f"Stage 4 output directory already exists: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 4 Qwen Adapter experiments require a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage 4 freezes bfloat16 compute, but this GPU lacks BF16 support")
    config = load_stage4_run_config(args.config)
    train_records = load_instruction_jsonl(Path(config.data.train_path))
    validation_records = load_instruction_jsonl(Path(config.data.validation_path))
    task_records = load_instruction_jsonl(Path(config.data.task_test_path))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_dir / "provenance.json", build_run_provenance(config))
    torch.cuda.reset_peak_memory_stats()
    print("stage=load_model", flush=True)
    stack = load_stage4_hf_stack(config)
    assert_stack_ready(stack)
    load_peak_bytes = int(torch.cuda.max_memory_allocated())
    train_examples = tokenize_records(
        train_records, stack.tokenizer, max_length=config.training.max_length
    )
    validation_examples = tokenize_records(
        validation_records, stack.tokenizer, max_length=config.training.max_length
    )
    print("stage=evaluate_before", flush=True)
    before = evaluate_stage4(stack, config, validation_examples, task_records)
    _write_json(args.output_dir / "before_training.json", before)
    print("stage=train", flush=True)
    training = train_bounded(stack, config, train_examples)
    print("stage=evaluate_after", flush=True)
    after = evaluate_stage4(stack, config, validation_examples, task_records)
    print("stage=save_adapter", flush=True)
    adapter_dir = args.output_dir / "adapter"
    stack.model.save_pretrained(adapter_dir, safe_serialization=True)
    stack.tokenizer.save_pretrained(adapter_dir)
    report = {
        "schema_version": "forgellm-stage4-hf-experiment-v1",
        "claim_boundary": (
            "Bounded single-configuration engineering evidence; not a benchmark or quality claim."
        ),
        "run_name": config.training.run_name,
        "model": {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "quantization": config.model.quantization,
            "total_parameters": stack.total_parameters,
            "trainable_parameters": stack.trainable_parameters,
            "trainable_fraction": stack.trainable_parameters / stack.total_parameters,
            "quantized_parameters": stack.quantized_parameters,
            "load_peak_allocated_bytes": load_peak_bytes,
            "initial_lora_noop_verified": True,
        },
        "training": training,
        "before_training": before,
        "after_training": after,
        "adapter_path": str(adapter_dir),
    }
    _write_json(report_path, report)
    print(report_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
