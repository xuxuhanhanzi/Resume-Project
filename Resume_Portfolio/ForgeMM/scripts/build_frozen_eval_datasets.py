"""Build answer-labelled frozen ChartQA and ChartQAPro ms-swift evaluation sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forgemm.data.chartqa import ChartQALoader  # noqa: E402

INSTRUCTION = (
    "Answer the chart question using the required evidence, operation, and answer XML blocks."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chartqa", type=Path, default=PROJECT_ROOT / "datasets/ChartQA")
    parser.add_argument(
        "--chartqapro",
        type=Path,
        default=PROJECT_ROOT / "datasets/ChartQAPro/chartqapro_test.parquet",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"schema_version": 1, "datasets": {}}
    for split in ("val", "test"):
        rows = _chartqa_rows(args.chartqa, split, root)
        target = output / f"chartqa_{split}_full.jsonl"
        _write_jsonl(target, rows)
        summary["datasets"][f"chartqa_{split}"] = _identity(target, len(rows))

    pro_target = output / "chartqapro_test_full.jsonl"
    pro_rows, invalid = _chartqapro_rows(args.chartqapro, output / "chartqapro_images", root)
    _write_jsonl(pro_target, pro_rows)
    summary["datasets"]["chartqapro_test"] = {
        **_identity(pro_target, len(pro_rows)),
        "invalid_questions_excluded": invalid,
        "image_files": len({row["images"][0] for row in pro_rows}),
    }
    summary_path = output / "frozen_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _chartqa_rows(chartqa: Path, split: str, root: Path) -> list[dict[str, Any]]:
    rows = []
    loader = ChartQALoader(chartqa)
    for source in ("human", "augmented"):
        for record in loader.iter_records(split, source):
            rows.append(
                _row(
                    record.record_id,
                    record.question,
                    record.answer,
                    _relative(record.image_path, root),
                    source=source,
                )
            )
    return rows


def _chartqapro_rows(
    parquet_path: Path, image_dir: Path, root: Path
) -> tuple[list[dict[str, Any]], int]:
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    invalid = 0
    table = pq.read_table(parquet_path)
    for row_index, row in enumerate(table.to_pylist()):
        image = row.get("image")
        if not isinstance(image, bytes):
            raise ValueError(f"invalid_chartqapro_image:{row_index}")
        image_path = image_dir / f"{row_index:06d}.jpg"
        if image_path.exists():
            if image_path.read_bytes() != image:
                raise ValueError(f"chartqapro_image_mismatch:{row_index}")
        else:
            image_path.write_bytes(image)
        questions = list(row.get("Question") or ())
        answers = list(row.get("Answer") or ())
        if len(questions) != len(answers):
            invalid += max(len(questions), len(answers))
            continue
        for question_index, (question, answer) in enumerate(zip(questions, answers, strict=True)):
            if question is None or answer is None or not str(question) or not str(answer):
                invalid += 1
                continue
            rows.append(
                _row(
                    f"chartqapro:test:{row_index}:{question_index}",
                    str(question),
                    str(answer),
                    _relative(image_path, root),
                    source="chartqapro",
                    question_type=str(row.get("Question Type") or "unknown"),
                )
            )
    return rows, invalid


def _row(
    sample_id: str,
    question: str,
    answer: str,
    image: str,
    *,
    source: str,
    question_type: str = "unknown",
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source": source,
        "question_type": question_type,
        "messages": [
            {"role": "user", "content": f"{INSTRUCTION}\nQuestion: {question}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [image],
    }


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"dataset_exists:{path}")
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _identity(path: Path, rows: int) -> dict[str, Any]:
    return {
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
