"""Run controlled tiny-model full-sequence, assistant-only and LoRA SFT experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from forgellm.model.config import ModelConfig
from forgellm.post_training.chat_template import tokenize_byte_bpe_record
from forgellm.post_training.method_lab import create_frozen_tiny_state, run_tiny_variant
from forgellm.post_training.schema import load_instruction_jsonl
from forgellm.structured_logging import JsonValue
from forgellm.tokenization.bpe import ByteBPETokenizer


def main(argv: Sequence[str] | None = None) -> int:
    """Execute low-cost three-seed method comparisons and one overfit gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"method report already exists: {args.output}")
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    records = load_instruction_jsonl(args.train)[:8]
    examples = [tokenize_byte_bpe_record(record, tokenizer, max_length=256) for record in records]
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=192,
        max_seq_len=256,
        attention_backend="sdpa",
    )
    results: list[dict[str, JsonValue]] = []
    for seed in (41, 42, 43):
        base_state = create_frozen_tiny_state(config, seed=seed)
        for variant in ("full_sequence", "assistant_only", "lora"):
            result = run_tiny_variant(
                base_state=base_state,
                model_config=config,
                examples=examples,
                variant=variant,
                seed=seed,
                steps=args.steps,
                learning_rate=3e-3 if variant != "lora" else 1e-2,
                device=device,
            )
            results.append(result.as_dict())
    report = {
        "schema_version": "forgellm-stage4-tiny-method-lab-v1",
        "claim_boundary": "Tiny correctness/method evidence only; not chat-model quality.",
        "device": str(device),
        "model": config.as_dict(),
        "records": [record.record_id for record in records],
        "steps_per_variant": args.steps,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
