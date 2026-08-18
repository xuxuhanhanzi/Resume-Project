"""DPO training for P10 — preference learning from success/failure pairs.

Self-contained manual DPO (TRL 1.9 DPOTrainer is incompatible with torch 2.5.1:
it requires `FSDPModule` which torch.distributed.fsdp does not export). This
script implements the standard DPO loss directly with transformers + peft,
continuing from the SFT adapter.

Reference model = frozen base (adapter disabled via PEFT disable_adapter()).
Policy = base + LoRA adapter (initialized from SFT checkpoint, trainable).

DPO loss per sample:
    chosen_reward   = beta * (logp_pol_chosen   - logp_ref_chosen)
    rejected_reward = beta * (logp_pol_rejected - logp_ref_rejected)
    loss = -logsigmoid(chosen_reward - rejected_reward)

Usage:
    python scripts/train_qwen_dpo.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as TorchDevice
    from transformers import PreTrainedModel

PROJECT = Path(__file__).resolve().parents[1]
DPO_DATA = PROJECT / "artifacts" / "training" / "p10_frames" / "dpo_data.jsonl"
SFT_ADAPTER = PROJECT / "artifacts" / "training" / "p10_qlora" / "adapter"
OUTPUT_DIR = PROJECT / "artifacts" / "training" / "p10_dpo"


def load_dpo_data(path: Path) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = (
                f"Question:\n{item['instruction']}\n\nSources:\n{item['input'][:4000]}\n\nAnswer:\n"
            )
            samples.append(
                {
                    "prompt": prompt,
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                }
            )
    return samples


def seq_logprob(
    model: PreTrainedModel,
    input_ids: list[int],
    comp_len: int,
    device: str | TorchDevice,
) -> Tensor:
    """Sum of token log-probs over the completion tokens.

    input_ids = prompt_ids + completion_ids (completion already ends with eos).
    comp_len = len(completion_ids). We score the completion tokens.
    """
    import torch

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    # labels: -100 for prompt, real id for completion tokens
    labels = ids.clone()
    prompt_len = len(input_ids) - comp_len
    labels[0, :prompt_len] = -100

    out = model(ids)
    logits = out.logits[:, :-1, :]
    targets = labels[:, 1:]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    # gather log-prob of target tokens where label != -100
    mask = targets != -100
    safe_targets = targets.clone()
    safe_targets[~mask] = 0
    token_lp = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (token_lp * mask).sum(dim=-1)  # [1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=1536)
    parser.add_argument("--init-from-sft", action="store_true", default=True)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("=== P10 DPO Training (manual impl, TRL-incompatible fallback) ===")
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

    if args.init_from_sft and SFT_ADAPTER.exists():
        print(f"Loading SFT adapter from {SFT_ADAPTER}")
        model = PeftModel.from_pretrained(model, str(SFT_ADAPTER), is_trainable=True)
    else:
        print("No SFT adapter found or disabled; DPO from base (trainable LoRA).")
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    model.train()
    device = next(model.parameters()).device

    data = load_dpo_data(DPO_DATA)
    print(f"Training data: {len(data)} DPO pairs")

    def build_ids(prompt: str, completion: str) -> tuple[list[int], int]:
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        c_ids = tokenizer(completion, add_special_tokens=False).input_ids
        c_ids = c_ids + [tokenizer.eos_token_id]
        full = p_ids + c_ids
        if len(full) > args.max_len:
            overflow = len(full) - args.max_len
            p_ids = p_ids[overflow:]
            full = p_ids + c_ids
        return full, len(c_ids)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    started = time.time()
    step = 0
    for epoch in range(args.epochs):
        for i, sample in enumerate(data):
            chosen_ids, chosen_len = build_ids(sample["prompt"], sample["chosen"])
            rejected_ids, rejected_len = build_ids(sample["prompt"], sample["rejected"])

            # policy logps (grad on)
            pol_chosen_lp = seq_logprob(model, chosen_ids, chosen_len, device)
            pol_rejected_lp = seq_logprob(model, rejected_ids, rejected_len, device)

            # reference logps (adapter disabled, no grad)
            with model.disable_adapter(), torch.no_grad():
                ref_chosen_lp = seq_logprob(model, chosen_ids, chosen_len, device)
                ref_rejected_lp = seq_logprob(model, rejected_ids, rejected_len, device)

            chosen_reward = args.beta * (pol_chosen_lp - ref_chosen_lp)
            rejected_reward = args.beta * (pol_rejected_lp - ref_rejected_lp)
            loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
            (loss / args.grad_accum).backward()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(data):
                torch.nn.utils.clip_grad_norm_(trainable, 0.3)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                torch.cuda.empty_cache()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(data):
                print(
                    f"epoch {epoch + 1}/{args.epochs} "
                    f"sample {i + 1}/{len(data)} "
                    f"loss={loss.item():.4f} "
                    f"chosen_rew={chosen_reward.item():.3f} "
                    f"rej_rew={rejected_reward.item():.3f} "
                    f"vram={torch.cuda.memory_allocated() / 1e9:.2f}GB",
                    flush=True,
                )

    elapsed = time.time() - started
    adapter_dir = OUTPUT_DIR / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    log = {
        "base_model": args.base_model,
        "method": "DPO (manual, TRL-incompatible fallback)",
        "init_from_sft": args.init_from_sft and SFT_ADAPTER.exists(),
        "beta": args.beta,
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_accum": args.grad_accum,
        "train_pairs": len(data),
        "optimizer_steps": step,
        "elapsed_seconds": round(elapsed, 1),
        "gpu": torch.cuda.get_device_name(0),
        "reference": "frozen base (adapter disabled)",
        "adapter_path": str(adapter_dir),
    }
    log_path = OUTPUT_DIR / "training_log.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nDPO training complete in {elapsed:.1f}s ({step} optimizer steps)")
    print(f"Adapter: {adapter_dir}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    import traceback

    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        err_path = OUTPUT_DIR / "dpo_error.log"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise SystemExit(1) from None
