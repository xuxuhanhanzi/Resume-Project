"""Traceable ms-swift SFT/GRPO dataset builders from strict EvidenceStore labels."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from forgemm.data.evidence_store import EvidenceStore, gold_evidence
from forgemm.data.schemas import EvidenceCell, EvidenceRecord, Operation

SFT_TEMPLATES = (
    "Answer the chart question using the required evidence, operation, and answer XML blocks.",
    "Act as a chart verifier: cite only supporting cells, execute one allowed operation, "
    "then answer.",
)
ANSWER_SFT_TEMPLATES = (
    "Answer the chart question concisely.",
    "Read the chart and provide only the final answer.",
)


def iter_strict_records(paths: Iterable[str | Path]) -> Iterable[EvidenceRecord]:
    seen: set[str] = set()
    for path in paths:
        for record in EvidenceStore(path).iter_records():
            if record.record_id in seen:
                raise ValueError(f"duplicate_record_id:{record.record_id}")
            seen.add(record.record_id)
            if (
                record.exclusion_reason is None
                and record.evidence_mask
                and record.operation_mask
                and record.gold_operation is not None
                and record.gold_evidence_ids
            ):
                yield record


def build_sft_rows(records: Iterable[EvidenceRecord]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        answer = render_structured_answer(record)
        for template_index, instruction in enumerate(SFT_TEMPLATES):
            rows.append(
                {
                    "sample_id": f"{record.record_id}:template-{template_index}",
                    "group_key": record.record_id,
                    "template_id": template_index,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{instruction}\nQuestion: {record.question}",
                        },
                        {"role": "assistant", "content": answer},
                    ],
                    "images": [record.image_path],
                }
            )
    return rows


def build_answer_sft_rows(records: Iterable[EvidenceRecord]) -> list[dict[str, Any]]:
    """Build an answer-only SFT control with the same strict-record population.

    The two views mirror the structured control's template count so that the
    E1/E2 comparison does not accidentally change the number of sampled records.
    """

    rows = []
    for record in records:
        for template_index, instruction in enumerate(ANSWER_SFT_TEMPLATES):
            rows.append(
                {
                    "sample_id": f"{record.record_id}:answer-template-{template_index}",
                    "group_key": record.record_id,
                    "template_id": template_index,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{instruction}\nQuestion: {record.question}",
                        },
                        {"role": "assistant", "content": record.reference_answer},
                    ],
                    "images": [record.image_path],
                }
            )
    return rows


def build_strict_eval_rows(records: Iterable[EvidenceRecord]) -> list[dict[str, Any]]:
    """Build one frozen structured target per strict question-level record."""

    rows = []
    for record in records:
        rows.append(
            {
                "sample_id": record.record_id,
                "source": record.source,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{SFT_TEMPLATES[0]}\nQuestion: {record.question}",
                    },
                    {"role": "assistant", "content": render_structured_answer(record)},
                ],
                "images": [record.image_path],
            }
        )
    return rows


def build_grpo_rows(records: Iterable[EvidenceRecord]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        cells = gold_evidence(record)
        rows.append(
            {
                "sample_id": record.record_id,
                "group_key": record.record_id,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{SFT_TEMPLATES[0]}\nQuestion: {record.question}",
                    }
                ],
                "images": [record.image_path],
                "reference_answer": record.reference_answer,
                "gold_evidence": json.dumps([asdict(cell) for cell in cells], sort_keys=True),
                "evidence_mask": record.evidence_mask,
                "operation_mask": record.operation_mask,
            }
        )
    return rows


def render_structured_answer(record: EvidenceRecord) -> str:
    operation = record.gold_operation
    if operation is None:
        raise ValueError(f"missing_gold_operation:{record.record_id}")
    cells = gold_evidence(record)
    aliases = {cell.evidence_id: f"e{index}" for index, cell in enumerate(cells, start=1)}
    evidence_lines = "\n".join(_render_cell(cell, aliases[cell.evidence_id]) for cell in cells)
    operation_text = _render_operation(operation, aliases)
    return (
        f"<evidence>\n{evidence_lines}\n</evidence>\n"
        f"<operation>\n{operation_text}\n</operation>\n"
        f"<answer>\n{record.reference_answer}\n</answer>"
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"dataset_exists:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _render_cell(cell: EvidenceCell, alias: str) -> str:
    row = json.dumps(cell.row, ensure_ascii=False)
    column = json.dumps(cell.column, ensure_ascii=False)
    value = json.dumps(cell.value, ensure_ascii=False)
    return f"{alias}=cell(row={row}, column={column}, value={value})"


def _render_operation(operation: Operation, aliases: dict[str, str]) -> str:
    arguments = []
    for argument in operation.arguments:
        value = aliases[argument.value] if argument.is_reference else argument.value
        rendered = value if argument.is_reference else json.dumps(value, ensure_ascii=False)
        arguments.append(f"{argument.name}={rendered}")
    return f"{operation.name}({', '.join(arguments)})"
