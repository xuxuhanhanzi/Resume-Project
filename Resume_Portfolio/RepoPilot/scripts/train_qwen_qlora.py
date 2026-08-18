"""QLoRA SFT training for P10 — Qwen2.5-1.5B, 4-bit, LoRA adapter.

Usage (after installing torch+peft+trl):
  python scripts/train_qwen_qlora.py --base-model Qwen/Qwen2.5-1.5B

Produces:
  artifacts/training/p10_qlora/adapter/  — LoRA adapter weights
  artifacts/training/p10_qlora/training_log.json — training metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

PROJECT = Path(__file__).resolve().parents[1]
SFT_DATA = PROJECT / "artifacts" / "training" / "p10_frames" / "sft_data.jsonl"
OUTPUT_DIR = PROJECT / "artifacts" / "training" / "p10_qlora"


def load_sft_data(path: Path) -> Dataset:
    from datasets import Dataset

    samples: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = (
                f"Question:\n{item['instruction']}\n\n"
                f"Sources:\n{item['input'][:4000]}\n\n"
                f"Answer:\n{item['response']}"
            )
            samples.append({"text": text})
    return Dataset.from_list(samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    print("=== P10 QLoRA Training ===")
    print(f"Base model: {args.base_model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_sft_data(SFT_DATA)
    print(f"Training data: {len(dataset)} samples")

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoint"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        max_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    import time

    started = time.time()
    trainer.train()
    elapsed = time.time() - started

    adapter_dir = OUTPUT_DIR / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    log = {
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "train_samples": len(dataset),
        "elapsed_seconds": round(elapsed, 1),
        "gpu": torch.cuda.get_device_name(0),
        "adapter_path": str(adapter_dir),
    }
    log_path = OUTPUT_DIR / "training_log.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Adapter: {adapter_dir}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
