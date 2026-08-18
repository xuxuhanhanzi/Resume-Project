"""Run a reproducible, validation-only Qwen2.5-VL baseline on ChartQA."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from forgellm.multimodal.chartqa import chartqa_relaxed_correct, normalized_exact_correct

PROMPT_PREFIX = (
    "Answer the chart question using only the chart. "
    "Return only the final short answer without explanation.\nQuestion: "
)


def _load_records(data_root: Path, subset: str) -> list[dict[str, str]]:
    annotation_path = data_root / "val" / f"val_{subset}.json"
    records = cast(object, json.loads(annotation_path.read_text(encoding="utf-8")))
    required_keys = {"imgname", "query", "label"}
    if not isinstance(records, list) or any(
        not isinstance(record, dict) or not required_keys <= record.keys() for record in records
    ):
        raise ValueError(f"Unexpected ChartQA schema: {annotation_path}")
    return [
        {key: str(record[key]) for key in required_keys}
        for record in cast(list[dict[str, object]], records)
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset", choices=("human", "augmented"), default="human")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=401_408)
    return parser


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
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_token_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return cast(
        str,
        processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.limit <= 0 or args.start_index < 0:
        raise ValueError("--limit must be positive and --start-index must be non-negative")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("ForgeMM baseline requires a CUDA GPU")

    records = _load_records(args.data_root, args.subset)
    selected_records = records[args.start_index : args.start_index + args.limit]
    if len(selected_records) != args.limit:
        raise ValueError("Requested validation range exceeds the selected ChartQA subset")
    image_dir = args.data_root / "val" / "png"
    missing_images = [
        record["imgname"]
        for record in selected_records
        if not (image_dir / record["imgname"]).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"Missing validation images, first: {missing_images[0]}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    quantization = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    torch.cuda.reset_peak_memory_stats()
    processor: Any = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_path, local_files_only=True
    )
    model: Any = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        quantization_config=quantization,
        device_map="auto",
    )
    model.eval()

    prediction_path = args.output_dir / "predictions.jsonl"
    normalized_exact_count = 0
    chartqa_relaxed_count = 0
    with prediction_path.open("w", encoding="utf-8") as prediction_file:
        for index, record in enumerate(selected_records, start=args.start_index):
            image_path = image_dir / record["imgname"]
            prediction = _generate_answer(
                model=model,
                processor=processor,
                image_path=image_path,
                question=record["query"],
                max_new_tokens=args.max_new_tokens,
                max_pixels=args.max_pixels,
            )
            normalized_exact = normalized_exact_correct(prediction, record["label"])
            chartqa_relaxed = chartqa_relaxed_correct(prediction, record["label"])
            normalized_exact_count += int(normalized_exact)
            chartqa_relaxed_count += int(chartqa_relaxed)
            prediction_file.write(
                json.dumps(
                    {
                        "index": index,
                        "image": record["imgname"],
                        "question": record["query"],
                        "reference": record["label"],
                        "prediction": prediction,
                        "normalized_exact_correct": normalized_exact,
                        "chartqa_relaxed_correct": chartqa_relaxed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(
                f"index={index} exact={normalized_exact} relaxed={chartqa_relaxed} "
                f"prediction={prediction!r}",
                flush=True,
            )

    summary = {
        "schema_version": "forgemm-chartqa-baseline-v1",
        "claim_boundary": "Validation-only smoke baseline; not a final benchmark result.",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_path": str(args.model_path.resolve()),
        "data_root": str(args.data_root.resolve()),
        "split": "val",
        "subset": args.subset,
        "count": len(selected_records),
        "start_index": args.start_index,
        "prompt": PROMPT_PREFIX,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "image_max_pixels": args.max_pixels,
        "quantization": "nf4-4bit-fp16-compute",
        "normalized_exact_accuracy": normalized_exact_count / len(selected_records),
        "normalized_exact_correct_count": normalized_exact_count,
        "chartqa_relaxed_accuracy": chartqa_relaxed_count / len(selected_records),
        "chartqa_relaxed_correct_count": chartqa_relaxed_count,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "prediction_path": str(prediction_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print((args.output_dir / "summary.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
