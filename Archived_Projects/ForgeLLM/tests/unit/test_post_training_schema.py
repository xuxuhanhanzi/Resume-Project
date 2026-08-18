"""Stage 4 conversational Schema and JSONL evidence tests."""

import json
from pathlib import Path

import pytest

from forgellm.post_training.schema import (
    InstructionDataError,
    InstructionRecord,
    Message,
    dataset_fingerprint,
    load_instruction_jsonl,
    write_instruction_jsonl,
)


def _record(record_id: str = "sample") -> InstructionRecord:
    return InstructionRecord(
        record_id=record_id,
        messages=(
            Message("system", "Answer exactly."),
            Message("user", f"Return identifier {record_id}."),
            Message("assistant", record_id),
        ),
        source="forgellm-stage4-correctness",
        license="project-original",
        metadata={"expected_response": record_id},
    )


def test_instruction_jsonl_round_trip_and_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    records = [_record("a"), _record("b")]

    write_instruction_jsonl(path, records)
    restored = load_instruction_jsonl(path)

    assert restored == records
    assert len(dataset_fingerprint(restored)) == 64
    assert restored[0].content_fingerprint() != restored[1].content_fingerprint()


@pytest.mark.parametrize(
    "messages",
    [
        (Message("assistant", "wrong start"),),
        (Message("user", "missing response"),),
        (Message("user", "one"), Message("user", "two"), Message("assistant", "three")),
        (Message("user", "one"), Message("assistant", "two"), Message("system", "late")),
    ],
)
def test_instruction_record_rejects_invalid_role_order(messages: tuple[Message, ...]) -> None:
    with pytest.raises(InstructionDataError):
        InstructionRecord("bad", messages, "fixture", "project-original")


def test_loader_rejects_duplicate_conversation_content(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.jsonl"
    first = _record("same")
    duplicate = InstructionRecord(
        record_id="different-id",
        messages=first.messages,
        source=first.source,
        license=first.license,
    )
    path.write_text(
        "\n".join(json.dumps(record.as_dict(), ensure_ascii=False) for record in (first, duplicate))
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InstructionDataError, match="duplicate conversation"):
        load_instruction_jsonl(path)


def test_writer_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_instruction_jsonl(path, [_record()])

    with pytest.raises(FileExistsError):
        write_instruction_jsonl(path, [_record("other")])
