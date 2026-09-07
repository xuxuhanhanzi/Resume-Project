"""Build the answer-only SFT control from the frozen strict training records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgemm.data.training_builders import (  # noqa: E402
    build_answer_sft_rows,
    iter_strict_records,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the answer-only SFT control dataset")
    parser.add_argument("--evidence-store", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--portable-image-root")
    args = parser.parse_args()

    records = list(iter_strict_records(args.evidence_store))
    if args.portable_image_root:
        root = Path(args.portable_image_root).resolve()
        records = [
            replace(record, image_path=_portable_image_path(record.image_path, root))
            for record in records
        ]
    rows = build_answer_sft_rows(records)
    target = Path(args.output)
    count = write_jsonl(target, rows)
    summary = {
        "schema_version": 1,
        "unique_strict_records": len(records),
        "sft_template_views_per_record": 2,
        "answer_sft_sequences": count,
        "sha256": _sha256(target),
        "image_path_mode": "project_relative" if args.portable_image_root else "source_record",
        "claim_boundary": "template views are not independent unique QA records",
    }
    summary_path = Path(args.summary)
    if summary_path.exists():
        raise FileExistsError(f"summary_exists:{summary_path}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _portable_image_path(source: str, project_root: Path) -> str:
    normalized = source.replace("\\", "/")
    marker = "/datasets/"
    marker_index = normalized.lower().find(marker)
    if marker_index < 0:
        raise ValueError(f"image_path_has_no_datasets_segment:{source}")
    relative = normalized[marker_index + 1 :]
    resolved = project_root.joinpath(*relative.split("/"))
    if not resolved.is_file():
        raise FileNotFoundError(f"rebased_image_missing:{resolved}")
    return relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
