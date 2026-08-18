"""Run bounded Answer-only SFT with a selected language-side QLoRA profile."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from PIL import Image
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from forgellm.multimodal.chartqa import chartqa_relaxed_correct, normalized_exact_correct
from forgellm.multimodal.sft import (
    assert_language_lora_only,
    deterministic_sample_index,
    language_lora_target_pattern,
    mask_prompt_labels,
)
from forgellm.post_training.adapters import adapter_base_gradients_are_absent
from forgellm.post_training.sft import scale_gradients_by_token_count

PROMPT_PREFIX = (
    "Answer the chart question using only the chart. "
    "Return only the final short answer without explanation.\nQuestion: "
)
LORA_PROFILES = {
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "attention-ffn": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_records(data_root: Path, split: str) -> list[dict[str, str]]:
    path = data_root / split / f"{split}_human.json"
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    required = {"imgname", "query", "label"}
    if not isinstance(payload, list) or any(
        not isinstance(record, dict) or not required <= record.keys() for record in payload
    ):
        raise ValueError(f"Unexpected ChartQA schema: {path}")
    return [
        {key: str(record[key]) for key in required}
        for record in cast(list[dict[str, object]], payload)
    ]


def _user_message(image_path: Path, question: str) -> dict[str, object]:
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": f"{PROMPT_PREFIX}{question}"},
        ],
    }


def _encode_answer_only(
    processor: Any,
    *,
    image_path: Path,
    question: str,
    answer: str,
    max_pixels: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    user_message = _user_message(image_path, question)
    prefix_text = processor.apply_chat_template(
        [user_message], tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        [
            user_message,
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        prefix = processor(
            text=[prefix_text],
            images=[rgb_image],
            return_tensors="pt",
            max_pixels=max_pixels,
        )
        complete = processor(
            text=[full_text],
            images=[rgb_image],
            return_tensors="pt",
            max_pixels=max_pixels,
        )
    prefix_length = int(prefix.input_ids.size(1))
    if not torch.equal(complete.input_ids[:, :prefix_length], prefix.input_ids):
        raise ValueError("Full multimodal sequence does not preserve the prompt prefix")
    batch = {key: value.to(device) for key, value in complete.items()}
    batch["labels"] = mask_prompt_labels(batch["input_ids"], prompt_length=prefix_length)
    return batch


def _generate(
    model: Any,
    processor: Any,
    *,
    image_path: Path,
    question: str,
    max_pixels: int,
    max_new_tokens: int,
) -> str:
    prompt = processor.apply_chat_template(
        [_user_message(image_path, question)], tokenize=False, add_generation_prompt=True
    )
    with Image.open(image_path) as image:
        inputs = processor(
            text=[prompt],
            images=[image.convert("RGB")],
            return_tensors="pt",
            max_pixels=max_pixels,
        ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer_ids = generated[:, inputs.input_ids.size(1) :]
    return cast(
        str,
        processor.batch_decode(
            answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip(),
    )


def _evaluate(
    model: Any,
    processor: Any,
    *,
    records: list[dict[str, str]],
    image_dir: Path,
    max_pixels: int,
    max_new_tokens: int,
) -> dict[str, object]:
    model.eval()
    predictions: list[dict[str, object]] = []
    exact_count = 0
    relaxed_count = 0
    for index, record in enumerate(records):
        prediction = _generate(
            model,
            processor,
            image_path=image_dir / record["imgname"],
            question=record["query"],
            max_pixels=max_pixels,
            max_new_tokens=max_new_tokens,
        )
        exact = normalized_exact_correct(prediction, record["label"])
        relaxed = chartqa_relaxed_correct(prediction, record["label"])
        exact_count += int(exact)
        relaxed_count += int(relaxed)
        predictions.append(
            {
                "index": index,
                "image": record["imgname"],
                "question": record["query"],
                "reference": record["label"],
                "prediction": prediction,
                "normalized_exact_correct": exact,
                "chartqa_relaxed_correct": relaxed,
            }
        )
    return {
        "count": len(records),
        "normalized_exact_accuracy": exact_count / len(records),
        "chartqa_relaxed_accuracy": relaxed_count / len(records),
        "predictions": predictions,
    }


def _frozen_run_config(args: argparse.Namespace) -> dict[str, object]:
    """Return settings that must remain unchanged across resume."""
    return {
        "model_path": str(args.model_path.resolve()),
        "data_root": str(args.data_root.resolve()),
        "train_limit": args.train_limit,
        "eval_limit": args.eval_limit,
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "target_profile": args.target_profile,
        "rank": args.rank,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }


def _build_scheduler(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a warmup plus cosine-decay scheduler with a 10% learning-rate floor."""

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _save_checkpoint(
    *,
    output_dir: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    trainer_state: dict[str, object],
) -> Path:
    """Save one immutable optimizer-boundary checkpoint."""
    optimizer_step = int(cast(int, trainer_state["optimizer_step"]))
    checkpoint_dir = output_dir / "checkpoints" / f"step_{optimizer_step:06d}"
    if checkpoint_dir.exists():
        return checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint_dir / "adapter", safe_serialization=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),  # type: ignore[no-untyped-call]
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
        checkpoint_dir / "training_state.pt",
    )
    _write_json(checkpoint_dir / "trainer_state.json", trainer_state)
    _write_json(output_dir / "latest_checkpoint.json", {"path": str(checkpoint_dir.resolve())})
    return checkpoint_dir


def _load_checkpoint_state(
    checkpoint_dir: Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> dict[str, object]:
    """Restore optimizer, scheduler, RNG and JSON trainer state."""
    payload = torch.load(
        checkpoint_dir / "training_state.pt",
        # CUDA RNG states are serialized as CPU ByteTensors.  Loading the whole
        # payload onto CUDA makes torch.cuda.set_rng_state_all reject them.
        # Optimizer.load_state_dict moves its parameter-bound states as needed.
        map_location="cpu",
        weights_only=False,
    )
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])  # type: ignore[no-untyped-call]
    random.setstate(payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    state = cast(
        dict[str, object],
        json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8")),
    )
    return state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=2)
    parser.add_argument("--eval-limit", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--target-profile",
        choices=tuple(LORA_PROFILES),
        default="attention",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-pixels", type=int, default=401_408)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.resume_from is None and args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.output_dir}")
    if args.resume_from is not None and not args.output_dir.is_dir():
        raise FileNotFoundError("Resume requires the existing original output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("Language-side QLoRA training requires CUDA")
    positive = (
        args.train_limit,
        args.eval_limit,
        args.max_steps,
        args.gradient_accumulation_steps,
        args.rank,
    )
    if min(positive) <= 0:
        raise ValueError("limits, steps, accumulation and rank must be positive")
    if min(args.warmup_steps, args.eval_interval, args.checkpoint_interval) < 0:
        raise ValueError("warmup and intervals must be non-negative")
    if args.warmup_steps >= args.max_steps:
        raise ValueError("warmup steps must be smaller than max steps")
    target_step = args.max_steps if args.stop_after_step is None else args.stop_after_step
    if not 0 < target_step <= args.max_steps:
        raise ValueError("stop-after-step must be within the configured training range")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    frozen_config = _frozen_run_config(args)
    lora_targets = LORA_PROFILES[args.target_profile]
    peft_target_pattern = language_lora_target_pattern(lora_targets)
    if args.resume_from is None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(args.output_dir / "run_config.json", frozen_config)
    else:
        existing_config = json.loads(
            (args.output_dir / "run_config.json").read_text(encoding="utf-8")
        )
        if existing_config != frozen_config:
            raise ValueError("Resume configuration differs from the original run")
    train_records = _load_records(args.data_root, "train")[: args.train_limit]
    eval_records = _load_records(args.data_root, "val")[: args.eval_limit]
    train_image_dir = args.data_root / "train" / "png"
    eval_image_dir = args.data_root / "val" / "png"

    quantization = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    torch.cuda.reset_peak_memory_stats()
    processor: Any = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_path, local_files_only=True
    )
    base: Any = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        quantization_config=quantization,
        device_map="auto",
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if args.resume_from is None:
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.0,
            target_modules=peft_target_pattern,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model: Any = get_peft_model(base, lora_config)
    else:
        model = PeftModel.from_pretrained(
            base,
            args.resume_from / "adapter",
            is_trainable=True,
        )
    trainable_names = assert_language_lora_only(
        (name for name, parameter in model.named_parameters() if parameter.requires_grad),
        target_modules=lora_targets,
    )
    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()

    if args.resume_from is None:
        before = _evaluate(
            model,
            processor,
            records=eval_records,
            image_dir=eval_image_dir,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
        )
        _write_json(args.output_dir / "evaluation_before.json", before)
    else:
        before = cast(
            dict[str, object],
            json.loads((args.output_dir / "evaluation_before.json").read_text(encoding="utf-8")),
        )

    trainable_parameters_list = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters_list,
        lr=args.learning_rate,
    )
    scheduler = _build_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
    )
    if args.resume_from is None:
        trainer_state: dict[str, object] = {
            "optimizer_step": 0,
            "micro_step": 0,
            "losses": [],
            "supervised_tokens_per_step": [],
            "learning_rates": [],
            "sample_indices": [],
            "evaluation_history": [],
            "best_exact_accuracy": float(cast(float, before["normalized_exact_accuracy"])),
            "best_adapter_path": None,
            "best_optimizer_step": 0,
        }
    else:
        trainer_state = _load_checkpoint_state(
            args.resume_from,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    while int(cast(int, trainer_state["optimizer_step"])) < target_step:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_nll = 0.0
        accumulated_tokens = 0
        step_sample_indices: list[int] = []
        for _ in range(args.gradient_accumulation_steps):
            micro_step = int(cast(int, trainer_state["micro_step"]))
            sample_index = deterministic_sample_index(
                len(train_records), micro_step, seed=args.seed
            )
            record = train_records[sample_index]
            batch = _encode_answer_only(
                processor,
                image_path=train_image_dir / record["imgname"],
                question=record["query"],
                answer=record["label"],
                max_pixels=args.max_pixels,
                device=model.device,
            )
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite QLoRA loss at micro step {micro_step}: {loss}")
            target_tokens = int(batch["labels"].ne(-100).sum().item())
            (loss * target_tokens).backward()
            accumulated_nll += float(loss.detach().item()) * target_tokens
            accumulated_tokens += target_tokens
            step_sample_indices.append(sample_index)
            trainer_state["micro_step"] = micro_step + 1

        scale_gradients_by_token_count(trainable_parameters_list, accumulated_tokens)
        if not adapter_base_gradients_are_absent(model):
            raise RuntimeError("A frozen base parameter acquired a gradient")
        nn.utils.clip_grad_norm_(trainable_parameters_list, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer_step = int(cast(int, trainer_state["optimizer_step"])) + 1
        trainer_state["optimizer_step"] = optimizer_step
        cast(list[float], trainer_state["losses"]).append(accumulated_nll / accumulated_tokens)
        cast(list[int], trainer_state["supervised_tokens_per_step"]).append(accumulated_tokens)
        cast(list[float], trainer_state["learning_rates"]).append(
            float(optimizer.param_groups[0]["lr"])
        )
        cast(list[list[int]], trainer_state["sample_indices"]).append(step_sample_indices)
        print(
            f"step={optimizer_step} loss="
            f"{cast(list[float], trainer_state['losses'])[-1]:.6f} "
            f"tokens={accumulated_tokens}",
            flush=True,
        )

        should_evaluate = args.eval_interval > 0 and optimizer_step % args.eval_interval == 0
        if should_evaluate:
            evaluation = _evaluate(
                model,
                processor,
                records=eval_records,
                image_dir=eval_image_dir,
                max_pixels=args.max_pixels,
                max_new_tokens=args.max_new_tokens,
            )
            evaluation["optimizer_step"] = optimizer_step
            _write_json(args.output_dir / f"evaluation_step_{optimizer_step:06d}.json", evaluation)
            cast(list[dict[str, object]], trainer_state["evaluation_history"]).append(
                {key: value for key, value in evaluation.items() if key != "predictions"}
            )
            exact_accuracy = float(cast(float, evaluation["normalized_exact_accuracy"]))
            current_best = trainer_state["best_exact_accuracy"]
            if current_best is None or exact_accuracy > float(cast(float, current_best)):
                best_dir = args.output_dir / "best_adapters" / f"step_{optimizer_step:06d}"
                model.save_pretrained(best_dir, safe_serialization=True)
                processor.save_pretrained(best_dir)
                trainer_state["best_exact_accuracy"] = exact_accuracy
                trainer_state["best_adapter_path"] = str(best_dir.resolve())
                trainer_state["best_optimizer_step"] = optimizer_step
            model.train()

        should_checkpoint = (
            args.checkpoint_interval > 0 and optimizer_step % args.checkpoint_interval == 0
        )
        if should_checkpoint:
            _save_checkpoint(
                output_dir=args.output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                trainer_state=trainer_state,
            )

    checkpoint_dir = _save_checkpoint(
        output_dir=args.output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state=trainer_state,
    )
    if target_step < args.max_steps:
        _write_json(
            args.output_dir / "interrupted.json",
            {
                "status": "intentional_stop",
                "optimizer_step": target_step,
                "resume_from": str(checkpoint_dir.resolve()),
            },
        )
        print(checkpoint_dir.resolve())
        return 0

    lora_b_tensors = [
        parameter
        for name, parameter in model.named_parameters()
        if "lora_B" in name and parameter.requires_grad
    ]
    nonzero_lora_b_tensors = sum(
        int(parameter.detach().count_nonzero().item() > 0) for parameter in lora_b_tensors
    )
    if not lora_b_tensors or nonzero_lora_b_tensors != len(lora_b_tensors):
        raise RuntimeError("Optimizer did not update every trainable LoRA-B tensor")

    final_evaluation_path = args.output_dir / f"evaluation_step_{args.max_steps:06d}.json"
    if final_evaluation_path.exists():
        after = cast(
            dict[str, object],
            json.loads(final_evaluation_path.read_text(encoding="utf-8")),
        )
    else:
        after = _evaluate(
            model,
            processor,
            records=eval_records,
            image_dir=eval_image_dir,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
        )
    _write_json(args.output_dir / "evaluation_after.json", after)
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)
    saved_adapter_config = PeftConfig.from_pretrained(adapter_dir)
    if saved_adapter_config.target_modules != peft_target_pattern:
        raise RuntimeError("Saved adapter target modules do not match the training boundary")

    report = {
        "schema_version": "forgemm-language-qlora-v4",
        "claim_boundary": "Bounded engineering run; not a quality or benchmark claim.",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_path": str(args.model_path.resolve()),
        "data_root": str(args.data_root.resolve()),
        "method": "answer-only-sft-nf4-qlora",
        "target_profile": args.target_profile,
        "target_modules": list(lora_targets),
        "peft_target_pattern": peft_target_pattern,
        "trainable_boundary": (
            "language_model.layers.*.{self_attn,mlp}.*.lora_*"
            if args.target_profile == "attention-ffn"
            else "language_model.layers.*.self_attn.*.lora_*"
        ),
        "trainable_parameter_names": len(trainable_names),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "parameter_count_method": "peft-4bit-aware",
        "saved_adapter_config_verified": True,
        "training_stack_before_definition": (
            "Initial zero-delta adapter after k-bit preparation; compare after-training here, "
            "and keep the original frozen inference baseline separate."
        ),
        "training": {
            "training_pool_examples": len(train_records),
            "unique_examples_seen": len(
                {
                    index
                    for indices in cast(list[list[int]], trainer_state["sample_indices"])
                    for index in indices
                }
            ),
            "max_steps": args.max_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_steps": args.warmup_steps,
            "rank": args.rank,
            "alpha": args.alpha,
            "learning_rate": args.learning_rate,
            "losses": trainer_state["losses"],
            "supervised_tokens_per_step": trainer_state["supervised_tokens_per_step"],
            "learning_rates": trainer_state["learning_rates"],
            "sample_indices": trainer_state["sample_indices"],
            "lora_b_tensors": len(lora_b_tensors),
            "nonzero_lora_b_tensors_after_training": nonzero_lora_b_tensors,
            "evaluation_history": trainer_state["evaluation_history"],
            "best_exact_accuracy": trainer_state["best_exact_accuracy"],
            "best_adapter_path": trainer_state["best_adapter_path"],
            "best_optimizer_step": trainer_state.get("best_optimizer_step"),
            "final_checkpoint_path": str(checkpoint_dir.resolve()),
        },
        "before": {key: value for key, value in before.items() if key != "predictions"},
        "after": {key: value for key, value in after.items() if key != "predictions"},
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "adapter_path": str(adapter_dir.resolve()),
    }
    _write_json(args.output_dir / "report.json", report)
    print((args.output_dir / "report.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
