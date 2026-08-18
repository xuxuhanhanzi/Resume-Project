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
    build_grpo_rows,
    build_sft_rows,
    iter_strict_records,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build traceable ms-swift datasets from strict labels"
    )
    parser.add_argument("--evidence-store", action="append", required=True)
    parser.add_argument("--sft-output", required=True)
    parser.add_argument("--grpo-output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--portable-image-root",
        help=(
            "verify each source image below this project root and emit a portable "
            "datasets/... path instead of the EvidenceStore absolute path"
        ),
    )
    args = parser.parse_args()

    records = list(iter_strict_records(args.evidence_store))
    if args.portable_image_root:
        root = Path(args.portable_image_root).resolve()
        records = [
            replace(record, image_path=_portable_image_path(record.image_path, root))
            for record in records
        ]
    sft_rows = build_sft_rows(records)
    grpo_rows = build_grpo_rows(records)
    sft_count = write_jsonl(args.sft_output, sft_rows)
    grpo_count = write_jsonl(args.grpo_output, grpo_rows)
    summary = {
        "schema_version": 1,
        "unique_strict_records": len(records),
        "sft_template_views_per_record": 2,
        "sft_sequences": sft_count,
        "grpo_prompts": grpo_count,
        "sft_minimum_3000_met": sft_count >= 3000,
        "grpo_minimum_1500_met": grpo_count >= 1500,
        "sft_sha256": _sha256(Path(args.sft_output)),
        "grpo_sha256": _sha256(Path(args.grpo_output)),
        "image_path_mode": "project_relative" if args.portable_image_root else "source_record",
        "claim_boundary": "template views are not independent unique QA records",
    }
    target = Path(args.summary)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


if __name__ == "__main__":
    raise SystemExit(main())
