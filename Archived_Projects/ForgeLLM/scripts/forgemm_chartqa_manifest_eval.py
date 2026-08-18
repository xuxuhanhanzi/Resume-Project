"""Evaluate one raw or PEFT-adapted Qwen-VL model on a CSV manifest with resume support."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from forgellm.multimodal.chartqa import chartqa_relaxed_correct, normalized_exact_correct
from forgellm.multimodal.evaluation import (
    file_sha256,
    load_completed_predictions,
    load_evaluation_manifest,
    summarize_predictions,
)

PROMPT_PREFIX = (
    "Answer the chart question using only the chart. "
    "Return only the final short answer without explanation.\nQuestion: "
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _generate_answer(
    *,
    model: Any,
    processor: Any,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    max_pixels: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": f"{PROMPT_PREFIX}{question}"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with Image.open(image_path) as image:
        inputs = processor(
            text=[prompt],
            images=[image.convert("RGB")],
            return_tensors="pt",
            padding=True,
            max_pixels=max_pixels,
        ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    answer_ids = generated[:, inputs.input_ids.shape[1] :]
    return cast(
        str,
        processor.batch_decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=401_408)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("Manifest evaluation requires CUDA")
    records = load_evaluation_manifest(args.manifest, limit=args.limit)
    image_dir = args.data_root / "val" / "png"
    missing = [record["image"] for record in records if not (image_dir / record["image"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation image: {missing[0]}")

    adapter_path = args.adapter_path.resolve() if args.adapter_path is not None else None
    adapter_weights = adapter_path / "adapter_model.safetensors" if adapter_path else None
    if adapter_weights is not None and not adapter_weights.is_file():
        raise FileNotFoundError(f"Adapter weights not found: {adapter_weights}")
    run_config = {
        "schema_version": "forgemm-manifest-eval-config-v1",
        "model_name": args.model_name,
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_sha256": file_sha256(adapter_weights) if adapter_weights else None,
        "data_root": str(args.data_root.resolve()),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "count": len(records),
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "prompt": PROMPT_PREFIX,
        "quantization": "nf4-4bit-fp16-compute",
        "inference_stack": "raw-4bit-base-with-optional-peft-adapter",
    }
    config_path = args.output_dir / "run_config.json"
    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume for existing run: {args.output_dir}")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise ValueError("Resume configuration differs from the original evaluation")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(config_path, run_config)

    prediction_path = args.output_dir / "predictions.jsonl"
    completed = load_completed_predictions(prediction_path, records)
    if len(completed) < len(records):
        quantization = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        torch.cuda.reset_peak_memory_stats()
        processor: Any = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            args.model_path,
            local_files_only=True,
        )
        base: Any = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            local_files_only=True,
            quantization_config=quantization,
            device_map="auto",
        )
        model: Any = (
            PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
            if adapter_path is not None
            else base
        )
        model.eval()
        started = time.perf_counter()
        with prediction_path.open("a", encoding="utf-8") as handle:
            for index in range(len(completed), len(records)):
                record = records[index]
                prediction = _generate_answer(
                    model=model,
                    processor=processor,
                    image_path=image_dir / record["image"],
                    question=record["question"],
                    max_new_tokens=args.max_new_tokens,
                    max_pixels=args.max_pixels,
                )
                row = {
                    "index": index,
                    "source_index": int(record["source_index"]),
                    "image": record["image"],
                    "task_type": record["task_type"],
                    "answer_type": record["answer_type"],
                    "question": record["question"],
                    "reference": record["reference"],
                    "prediction": prediction,
                    "normalized_exact_correct": normalized_exact_correct(
                        prediction, record["reference"]
                    ),
                    "chartqa_relaxed_correct": chartqa_relaxed_correct(
                        prediction, record["reference"]
                    ),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed.append(row)
                if (index + 1) % 10 == 0 or index + 1 == len(records):
                    print(
                        f"model={args.model_name} completed={index + 1}/{len(records)}",
                        flush=True,
                    )
        elapsed = time.perf_counter() - started
        peak_memory = int(torch.cuda.max_memory_allocated())
    else:
        elapsed = 0.0
        peak_memory = 0

    completed = load_completed_predictions(prediction_path, records)
    metrics = summarize_predictions(completed)
    summary = {
        "schema_version": "forgemm-manifest-eval-v1",
        "claim_boundary": "Single fixed manifest; heuristic category labels are not official.",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_config": run_config,
        "metrics": metrics,
        "last_invocation_elapsed_seconds": elapsed,
        "last_invocation_peak_cuda_allocated_bytes": peak_memory,
        "prediction_path": str(prediction_path.resolve()),
    }
    _write_json(args.output_dir / "summary.json", summary)
    print((args.output_dir / "summary.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
