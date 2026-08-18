"""Frozen case construction and bounded contamination preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from forgellm.alignment.schema import read_preference_jsonl
from forgellm.evaluation.cases import build_stage6_cases, write_cases
from forgellm.evaluation.config import Stage6Config
from forgellm.evaluation.contamination import audit_contamination, normalized_sha256
from forgellm.evaluation.identity import file_sha256
from forgellm.post_training.schema import Message, load_instruction_jsonl
from forgellm.structured_logging import JsonValue


def _messages_text(messages: tuple[Message, ...]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def _instruction_training_texts(path: Path) -> dict[str, str]:
    return {
        f"{path.parent.name}:{record.record_id}": _messages_text(record.messages)
        for record in load_instruction_jsonl(path)
    }


def _preference_training_texts(path: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for record in read_preference_jsonl(path):
        prefix = f"{path.parent.name}:{record.record_id}"
        prompt = _messages_text(record.prompt)
        texts[f"{prefix}:chosen"] = f"{prompt}\nassistant: {record.chosen}"
        texts[f"{prefix}:rejected"] = f"{prompt}\nassistant: {record.rejected}"
    return texts


def _retention_exact_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = cast(object, json.loads(line))
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            raise ValueError(f"invalid retention record in {path}")
        hashes.add(normalized_sha256(cast(str, raw["text"])))
    return hashes


def prepare_stage6_data(config: Stage6Config, *, project_root: Path) -> dict[str, JsonValue]:
    """Create cases and contamination evidence while refusing to overwrite it."""
    paths = config.data
    cases_path = project_root / paths.cases_path
    correctness_test = project_root / paths.correctness_test_path
    preference_test = project_root / paths.preference_test_path
    cases = build_stage6_cases(correctness_test, preference_test)
    if len(cases) != 64:
        raise RuntimeError(f"frozen Stage 6 case count changed: {len(cases)}")
    write_cases(cases_path, cases)

    evaluation_texts = {
        case.case_id: f"{_messages_text(case.prompt)}\nassistant: {case.expected_response}"
        for case in cases
    }
    training_texts: dict[str, str] = {}
    training_texts.update(_instruction_training_texts(project_root / paths.correctness_train_path))
    training_texts.update(_instruction_training_texts(project_root / paths.smoltalk_train_path))
    training_texts.update(_preference_training_texts(project_root / paths.preference_train_path))
    matches = audit_contamination(
        evaluation_texts,
        training_texts,
        ngram_size=config.quality.near_duplicate_ngram,
        near_threshold=config.quality.near_duplicate_threshold,
    )
    retention_hashes = _retention_exact_hashes(project_root / paths.retention_train_path)
    retention_exact = sorted(
        case_id
        for case_id, text in evaluation_texts.items()
        if normalized_sha256(text) in retention_hashes
    )
    source_paths = {
        "correctness_train": paths.correctness_train_path,
        "correctness_test": paths.correctness_test_path,
        "preference_train": paths.preference_train_path,
        "preference_test": paths.preference_test_path,
        "smoltalk_train": paths.smoltalk_train_path,
        "retention_train": paths.retention_train_path,
        "retention_test": paths.retention_test_path,
    }
    report: dict[str, JsonValue] = {
        "schema_version": "forgellm-stage6-preparation-v1",
        "case_count": len(cases),
        "case_types": {
            task: sum(case.task_type == task for case in cases)
            for task in ("correctness", "preference", "robustness")
        },
        "cases_sha256": file_sha256(cases_path),
        "source_data_sha256": {
            name: file_sha256(project_root / value) for name, value in source_paths.items()
        },
        "contamination": {
            "normalization": "NFKC + casefold + collapsed whitespace",
            "character_ngram": config.quality.near_duplicate_ngram,
            "near_duplicate_threshold": config.quality.near_duplicate_threshold,
            "bounded_structured_training_records": len(training_texts),
            "exact_matches": sum(item.exact for item in matches) + len(retention_exact),
            "near_matches": sum(not item.exact for item in matches),
            "structured_matches": cast(JsonValue, [item.as_dict() for item in matches]),
            "retention_exact_case_ids": cast(JsonValue, retention_exact),
            "interpretation": (
                "Template overlap is reported, not silently deleted; exact overlap is a hard flag."
            ),
        },
    }
    report_path = cases_path.with_name("preparation_report.json")
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
