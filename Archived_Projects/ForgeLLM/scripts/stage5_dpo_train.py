"""Run one immutable bounded Qwen DPO experiment from the Stage 4 SFT Adapter."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from forgellm.alignment.config import load_stage5_run_config
from forgellm.alignment.hf_common import (
    dropout_is_disabled,
    file_sha256,
    load_stage5_policy,
    seed_everything,
)
from forgellm.alignment.hf_dpo import (
    evaluate_preference_generation,
    evaluate_preference_log_probs,
    precompute_reference,
    reference_cache_as_json,
    train_dpo_bounded,
)
from forgellm.alignment.schema import read_preference_jsonl
from forgellm.post_training.adapters import HFStack
from forgellm.post_training.config import load_stage4_run_config
from forgellm.post_training.hf_experiment import evaluate_stage4, tokenize_records
from forgellm.post_training.schema import load_instruction_jsonl


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Stage 5 DPO output already exists: {args.output_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage 5 Qwen DPO requires a CUDA GPU with BF16 support")
    config = load_stage5_run_config(args.config)
    all_train_records = read_preference_jsonl(Path(config.data.train_path))
    train_limit = min(
        len(all_train_records),
        config.dpo.max_steps * config.dpo.gradient_accumulation_steps,
    )
    train_records = all_train_records[:train_limit]
    validation_records = read_preference_jsonl(Path(config.data.validation_path))[
        : config.dpo.evaluation_pairs
    ]
    test_records = read_preference_jsonl(Path(config.data.test_path))
    stage4_config = load_stage4_run_config(Path(config.data.stage4_evaluation_config))
    stage4_validation_records = load_instruction_jsonl(Path(stage4_config.data.validation_path))
    stage4_task_records = load_instruction_jsonl(Path(stage4_config.data.task_test_path))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(config.dpo.seed)
    provenance = {
        "schema_version": "forgellm-stage5-dpo-provenance-v1",
        "config": config.as_dict(),
        "config_fingerprint": config.fingerprint(),
        "input_sha256": {
            "train": file_sha256(Path(config.data.train_path)),
            "validation": file_sha256(Path(config.data.validation_path)),
            "test": file_sha256(Path(config.data.test_path)),
            "initial_adapter": file_sha256(
                Path(config.model.initial_adapter_path) / "adapter_model.safetensors"
            ),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("peft", "torch", "transformers", "trl")
        },
        "selected_training_records": len(train_records),
        "selected_validation_records": len(validation_records),
    }
    _write_json(args.output_dir / "provenance.json", provenance)
    torch.cuda.reset_peak_memory_stats()
    print("stage=load_sft_policy", flush=True)
    stack = load_stage5_policy(config, trainable=True)
    load_peak = int(torch.cuda.max_memory_allocated())
    print("stage=precompute_reference", flush=True)
    train_reference = precompute_reference(stack, train_records, max_length=config.dpo.max_length)
    validation_reference = precompute_reference(
        stack, validation_records, max_length=config.dpo.max_length
    )
    _write_json(
        args.output_dir / "reference_log_probs.json",
        {
            "adapter_sha256": stack.initial_adapter_sha256,
            "train": reference_cache_as_json(train_reference),
            "validation": reference_cache_as_json(validation_reference),
        },
    )
    stage4_stack = HFStack(
        stack.tokenizer,
        stack.model,
        stack.total_parameters,
        stack.trainable_parameters,
        0,
    )
    stage4_validation_examples = tokenize_records(
        stage4_validation_records,
        stack.tokenizer,
        max_length=stage4_config.training.max_length,
    )
    print("stage=evaluate_before", flush=True)
    before = {
        "preference_log_probs": evaluate_preference_log_probs(
            stack,
            validation_records,
            validation_reference,
            beta=config.dpo.beta,
            max_length=config.dpo.max_length,
            limit=config.dpo.evaluation_pairs,
        ),
        "preference_generation": evaluate_preference_generation(
            stack.model,
            stack.tokenizer,
            test_records,
            max_length=config.dpo.max_length,
            max_new_tokens=config.dpo.max_new_tokens,
            limit=config.dpo.generation_examples,
        ),
        "stage4_fixed_evaluation": evaluate_stage4(
            stage4_stack,
            stage4_config,
            stage4_validation_examples,
            stage4_task_records,
        ),
    }
    _write_json(args.output_dir / "before_training.json", before)
    print("stage=train_dpo", flush=True)
    training = train_dpo_bounded(stack, config, train_records, train_reference)
    print("stage=evaluate_after", flush=True)
    after = {
        "preference_log_probs": evaluate_preference_log_probs(
            stack,
            validation_records,
            validation_reference,
            beta=config.dpo.beta,
            max_length=config.dpo.max_length,
            limit=config.dpo.evaluation_pairs,
        ),
        "preference_generation": evaluate_preference_generation(
            stack.model,
            stack.tokenizer,
            test_records,
            max_length=config.dpo.max_length,
            max_new_tokens=config.dpo.max_new_tokens,
            limit=config.dpo.generation_examples,
        ),
        "stage4_fixed_evaluation": evaluate_stage4(
            stage4_stack,
            stage4_config,
            stage4_validation_examples,
            stage4_task_records,
        ),
    }
    adapter_dir = args.output_dir / "adapter"
    stack.model.save_pretrained(adapter_dir, safe_serialization=True)
    stack.tokenizer.save_pretrained(adapter_dir)
    adapter_sha = file_sha256(adapter_dir / "adapter_model.safetensors")
    report = {
        "schema_version": "forgellm-stage5-dpo-v1",
        "claim_boundary": (
            "One bounded verifier-pair DPO run; not a general preference benchmark or RL result."
        ),
        "run_name": config.dpo.run_name,
        "model": {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "total_parameters": stack.total_parameters,
            "trainable_parameters": stack.trainable_parameters,
            "load_peak_allocated_bytes": load_peak,
            "initial_adapter_sha256": stack.initial_adapter_sha256,
            "final_adapter_sha256": adapter_sha,
            "reference_adapter_sha256": stack.initial_adapter_sha256,
            "dropout_disabled": dropout_is_disabled(stack.model),
        },
        "training": training,
        "before_training": before,
        "after_training": after,
        "adapter_path": str(adapter_dir),
    }
    _write_json(args.output_dir / "report.json", report)
    print((args.output_dir / "report.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
