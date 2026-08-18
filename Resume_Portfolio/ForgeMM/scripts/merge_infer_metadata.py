"""Restore frozen dataset identifiers that ms-swift omits from inference JSONL output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_METADATA_KEYS = ("sample_id", "source", "question_type")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = _load(args.dataset)
    inference = _load(args.inference)
    merged = merge_rows(dataset, inference)
    if args.output.exists():
        raise FileExistsError(f"output_exists:{args.output}")
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in merged),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"rows": len(merged), "output": str(args.output)}, sort_keys=True))
    return 0


def merge_rows(
    dataset: list[dict[str, Any]], inference: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not dataset or len(dataset) != len(inference):
        raise ValueError("dataset and inference rows must be non-empty and aligned")
    result = []
    for index, (source, prediction) in enumerate(zip(dataset, inference, strict=True)):
        expected_labels = source["messages"][-1]["content"]
        if str(prediction.get("labels")) != str(expected_labels):
            raise ValueError(f"label_identity_mismatch:{index}")
        if _image_paths(prediction.get("images")) != _image_paths(source.get("images")):
            raise ValueError(f"row_identity_mismatch:{index}:images")
        result.append(
            {
                **prediction,
                **{key: source[key] for key in _METADATA_KEYS if key in source},
            }
        )
    return result


def _image_paths(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item.get("path", "")) if isinstance(item, dict) else str(item) for item in value]


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    raise SystemExit(main())
