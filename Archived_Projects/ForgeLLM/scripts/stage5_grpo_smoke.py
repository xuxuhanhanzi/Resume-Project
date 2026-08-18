"""Run the immutable four-prompt, group-four, one-step Qwen GRPO smoke."""

from __future__ import annotations

import argparse
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
from forgellm.alignment.hf_grpo import run_one_step_grpo
from forgellm.alignment.schema import read_preference_jsonl


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
        raise FileExistsError(f"Stage 5 GRPO output already exists: {args.output_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage 5 Qwen GRPO smoke requires CUDA BF16")
    config = load_stage5_run_config(args.config)
    records = read_preference_jsonl(Path(config.data.test_path))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(config.grpo.seed)
    _write_json(
        args.output_dir / "provenance.json",
        {
            "schema_version": "forgellm-stage5-grpo-provenance-v1",
            "config": config.as_dict(),
            "config_fingerprint": config.fingerprint(),
            "test_sha256": file_sha256(Path(config.data.test_path)),
            "initial_adapter_sha256": file_sha256(
                Path(config.model.initial_adapter_path) / "adapter_model.safetensors"
            ),
        },
    )
    torch.cuda.reset_peak_memory_stats()
    print("stage=load_sft_policy", flush=True)
    stack = load_stage5_policy(config, trainable=True)
    load_peak = int(torch.cuda.max_memory_allocated())
    print("stage=rollout_reward_update", flush=True)
    training = run_one_step_grpo(stack, config, records, output_dir=args.output_dir)
    adapter_dir = args.output_dir / "adapter"
    stack.model.save_pretrained(adapter_dir, safe_serialization=True)
    stack.tokenizer.save_pretrained(adapter_dir)
    final_adapter_sha = file_sha256(adapter_dir / "adapter_model.safetensors")
    report = {
        "schema_version": "forgellm-stage5-grpo-smoke-v1",
        "claim_boundary": (
            "One real on-policy update proving the local chain; no capability-improvement claim."
        ),
        "run_name": config.grpo.run_name,
        "model": {
            "model_id": config.model.model_id,
            "revision": config.model.revision,
            "total_parameters": stack.total_parameters,
            "trainable_parameters": stack.trainable_parameters,
            "load_peak_allocated_bytes": load_peak,
            "initial_adapter_sha256": stack.initial_adapter_sha256,
            "final_adapter_sha256": final_adapter_sha,
            "dropout_disabled": dropout_is_disabled(stack.model),
        },
        "training": training,
        "adapter_path": str(adapter_dir),
    }
    _write_json(args.output_dir / "report.json", report)
    print((args.output_dir / "report.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
