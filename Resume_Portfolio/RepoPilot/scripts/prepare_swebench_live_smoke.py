"""Extract public/private SWE-bench-Live smoke records from a pinned parquet file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

DEFAULT_TASK_IDS = (
    "aws-cloudformation__cfn-lint-3767",
    "python-babel__babel-1141",
    "projectmesa__mesa-2394",
)

PUBLIC_FIELDS = (
    "repo",
    "pull_number",
    "instance_id",
    "issue_numbers",
    "base_commit",
    "problem_statement",
    "created_at",
    "commit_url",
    "difficulty",
)

EVALUATOR_FIELDS = (
    "repo",
    "instance_id",
    "base_commit",
    "patch",
    "test_patch",
    "test_cmds",
    "log_parser",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expected-sha256", required=True)
    arguments = parser.parse_args()

    source = arguments.parquet.resolve(strict=True)
    if _sha256(source) != str(arguments.expected_sha256).casefold():
        raise ValueError("SWE-bench-Live parquet hash does not match the pinned source")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required only for this dataset preparation script"
        ) from error

    rows = cast(list[dict[str, Any]], parquet.read_table(source).to_pylist())
    by_id = {str(row["instance_id"]): row for row in rows}
    missing = [task_id for task_id in DEFAULT_TASK_IDS if task_id not in by_id]
    if missing:
        raise ValueError(f"pinned SWE-bench-Live split is missing smoke tasks: {missing}")

    public_records = [
        {field: by_id[task_id][field] for field in PUBLIC_FIELDS} for task_id in DEFAULT_TASK_IDS
    ]
    evaluator_records = [
        {field: by_id[task_id][field] for field in EVALUATOR_FIELDS} for task_id in DEFAULT_TASK_IDS
    ]
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    public_path = output_dir / "smoke_public.jsonl"
    evaluator_path = output_dir / "smoke_evaluator.jsonl"
    _write_jsonl(public_path, public_records)
    _write_jsonl(evaluator_path, evaluator_records)
    selection = {
        "dataset": "SWE-bench-Live/SWE-bench-Live",
        "revision": arguments.revision,
        "source_parquet": source.name,
        "source_sha256": _sha256(source),
        "task_ids": list(DEFAULT_TASK_IDS),
        "public_sha256": _sha256(public_path),
        "evaluator_sha256": _sha256(evaluator_path),
    }
    (output_dir / "smoke_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
