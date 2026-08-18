"""Build a conservative ChartQA EvidenceStore from a frozen audit report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forgemm.data.chartqa import ChartQALoader  # noqa: E402
from forgemm.data.evidence_store import (  # noqa: E402
    LABELER_VERSION,
    EvidenceStore,
    build_table_evidence,
    derive_supported_label,
)
from forgemm.data.schemas import EvidenceRecord  # noqa: E402
from forgemm.reasoning.normalizer import normalize_text  # noqa: E402

BUILDER_VERSION = "chartqa-store-1.1.0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chartqa", type=Path, default=PROJECT_ROOT / "datasets" / "ChartQA")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--source", choices=("human", "augmented"), default="human")
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    audit = _load_object(args.audit_report)
    excluded = _audit_exclusions(audit, args.split)
    loader = ChartQALoader(args.chartqa)
    templates: dict[str, EvidenceRecord] = {}
    seen_questions: set[tuple[str, str, str]] = set()
    records: list[EvidenceRecord] = []
    reasons = Counter[str]()
    for source in loader.iter_records(args.split, args.source):
        template = templates.get(source.image_name)
        if template is None:
            template = build_table_evidence(source)
            templates[source.image_name] = template
        record = replace(
            template,
            record_id=source.record_id,
            question=source.question,
            reference_answer=source.answer,
        )
        audit_reason = excluded.get(source.image_name)
        if audit_reason is not None:
            record = replace(record, evidence_mask=False, exclusion_reason=audit_reason)
        question_key = (
            source.image_name,
            normalize_text(source.question),
            normalize_text(source.answer),
        )
        if question_key in seen_questions:
            record = replace(
                record,
                evidence_mask=False,
                operation_mask=False,
                exclusion_reason="duplicate_qa_record",
            )
        seen_questions.add(question_key)
        record = derive_supported_label(record)
        record = replace(record, builder_version=BUILDER_VERSION)
        reasons[record.exclusion_reason or "eligible"] += 1
        records.append(record)

    count = EvidenceStore(args.output).write(records)
    summary = {
        "schema_version": "1.3.0",
        "dataset": "ChartQA",
        "split": args.split,
        "source": args.source,
        "records": count,
        "unique_charts": len(templates),
        "evidence_eligible": sum(record.evidence_mask for record in records),
        "operation_eligible": sum(record.operation_mask for record in records),
        "labeler_version": LABELER_VERSION,
        "builder_version": BUILDER_VERSION,
        "reason_counts": dict(sorted(reasons.items())),
        "audit_report": str(args.audit_report),
        "store": str(args.output),
    }
    if args.summary.exists():
        raise FileExistsError(f"summary_exists:{args.summary}")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected_object:{path}")
    return cast(dict[str, Any], payload)


def _audit_exclusions(audit: dict[str, Any], split: str) -> dict[str, str]:
    chartqa = cast(dict[str, Any], audit["chartqa"])
    splits = cast(dict[str, Any], chartqa["splits"])
    split_report = cast(dict[str, Any], splits[split])
    exclusions = {
        str(name): "corrupt_image"
        for name in cast(list[object], split_report.get("corrupt_image_examples", []))
    }
    duplicates = cast(list[dict[str, str]], chartqa.get("cross_split_duplicate_examples", []))
    for duplicate in duplicates:
        for location in duplicate["locations"].split(","):
            location_split, image_name = location.split("/", 1)
            if location_split == split:
                exclusions[image_name] = "cross_split_duplicate_image"
    return exclusions


if __name__ == "__main__":
    raise SystemExit(main())
