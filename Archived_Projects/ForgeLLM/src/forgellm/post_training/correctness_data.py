"""Deterministic project-original SFT fixtures with rule-verifiable answers."""

from __future__ import annotations

from forgellm.post_training.schema import InstructionRecord, Message
from forgellm.structured_logging import JsonValue


def build_correctness_records() -> tuple[InstructionRecord, ...]:
    """Create 96 unique records covering exact, word-count, JSON and multi-turn forms."""
    records: list[InstructionRecord] = []
    for index in range(96):
        marker = f"ITEM-{index:03d}"
        variant = index % 4
        metadata: dict[str, JsonValue]
        if variant == 0:
            prompt = f"Reply with exactly this identifier and nothing else: {marker}"
            response = marker
            metadata = {"expected_response": response}
        elif variant == 1:
            first = f"red{index}"
            second = f"blue{index}"
            prompt = f"Reply with exactly two space-separated words: {first} {second}"
            response = f"{first} {second}"
            metadata = {
                "exact_word_count": 2,
                "expected_response": response,
                "required_substrings": [first, second],
            }
        elif variant == 2:
            prompt = f'Output only JSON with key "id" and string value "{marker}".'
            response = f'{{"id":"{marker}"}}'
            metadata = {"expected_response": response, "require_json": True}
        else:
            prompt = f"Start the answer with RESULT and then write {marker}."
            response = f"RESULT {marker}"
            metadata = {
                "expected_response": response,
                "starts_with": "RESULT",
                "required_substrings": [marker],
            }
        messages: tuple[Message, ...]
        if index % 8 == 7:
            messages = (
                Message("system", "Follow the requested output format exactly."),
                Message("user", "Remember that extra commentary is forbidden."),
                Message("assistant", "Understood."),
                Message("user", prompt),
                Message("assistant", response),
            )
        else:
            messages = (
                Message("system", "Follow the requested output format exactly."),
                Message("user", prompt),
                Message("assistant", response),
            )
        records.append(
            InstructionRecord(
                record_id=f"correctness-{index:03d}",
                messages=messages,
                source="forgellm-stage4-correctness-v1",
                license="project-original",
                metadata=metadata,
            )
        )
    return tuple(records)


def split_correctness_records(
    records: tuple[InstructionRecord, ...],
) -> dict[str, tuple[InstructionRecord, ...]]:
    """Apply the frozen 64/16/16 split without random state."""
    if len(records) != 96:
        raise ValueError("sft_correctness_v1 must contain exactly 96 records")
    return {
        "train": records[:64],
        "validation": records[64:80],
        "test": records[80:],
    }
