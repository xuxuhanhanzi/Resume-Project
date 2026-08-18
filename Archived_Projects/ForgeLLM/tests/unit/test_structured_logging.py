"""Tests for structured logging and secret redaction."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from forgellm.structured_logging import redact, write_jsonl_event


def test_redact_nested_sensitive_fields() -> None:
    result = redact(
        {
            "authorization": "Bearer real-secret",
            "nested": {"api-key": "real-key", "safe": "visible"},
        }
    )

    assert result == {
        "authorization": "[REDACTED]",
        "nested": {"api-key": "[REDACTED]", "safe": "visible"},
    }


def test_llm_token_metrics_are_not_mistaken_for_credentials() -> None:
    assert redact({"tokenizer": "bpe", "prompt_tokens": 12, "token_budget": 100}) == {
        "tokenizer": "bpe",
        "prompt_tokens": 12,
        "token_budget": 100,
    }


def test_write_jsonl_event_is_stable_and_redacted(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    write_jsonl_event(
        path,
        "test_event",
        {"token": "do-not-log", "value": 3},
        now=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "event": "test_event",
        "timestamp": "2026-07-11T12:00:00Z",
        "token": "[REDACTED]",
        "value": 3,
    }


def test_reserved_log_fields_cannot_be_overwritten(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved keys: event"):
        write_jsonl_event(tmp_path / "events.jsonl", "safe", {"event": "spoofed"})
