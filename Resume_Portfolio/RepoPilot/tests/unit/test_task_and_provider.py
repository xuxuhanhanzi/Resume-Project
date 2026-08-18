from __future__ import annotations

import json
from pathlib import Path

import pytest

from repopilot.core.contracts import ModelRequest
from repopilot.providers.local_openai import LocalOpenAICompatibleProvider, LocalProviderConfig
from repopilot.providers.scripted import ScriptedProvider
from repopilot.task import EvaluatorTaskSpec, PublicTaskSpec


def test_public_task_rejects_hidden_tests(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    path.write_text(
        "task_id: t\nproblem_statement: fix it\nhidden_tests: [secret]\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="must not contain hidden_tests"):
        PublicTaskSpec.load(path)


def test_evaluator_task_keeps_hidden_tests_separate(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    path.write_text(
        "task_id: t\nproblem_statement: fix it\ntrusted_fixture: true\n"
        "evaluator:\n  hidden_tests: [hidden.py]\n",
        encoding="utf-8",
    )

    spec = EvaluatorTaskSpec.load(path)

    assert spec.public.task_id == "t"
    assert spec.hidden_tests == ("hidden.py",)


def test_task_commands_must_be_token_lists(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="test_command"):
        PublicTaskSpec.from_mapping(
            {"task_id": "t", "problem_statement": "fix", "test_command": "python verify.py"},
            base_dir=tmp_path,
        )


def test_local_provider_is_loopback_only() -> None:
    with pytest.raises(ValueError, match="loopback"):
        LocalProviderConfig("https://example.com", "model")


def test_local_provider_parses_native_and_json_tool_calls() -> None:
    provider = LocalOpenAICompatibleProvider(LocalProviderConfig("http://127.0.0.1:1", "qwen"))
    native = provider._parse_response(  # noqa: SLF001
        {
            "model": "qwen",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "a.py"}),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
    )
    fallback = provider._parse_response(  # noqa: SLF001
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "type": "tool_call",
                                "tool": "read_file",
                                "arguments": {"path": "b.py"},
                            }
                        )
                    }
                }
            ]
        }
    )

    assert native.tool_calls[0].call_id == "c1"
    assert native.usage.total_tokens == 14
    assert fallback.tool_calls[0].arguments == {"path": "b.py"}


def test_scripted_provider_records_requests() -> None:
    import asyncio

    from repopilot.core.contracts import ModelResponse

    provider = ScriptedProvider([ModelResponse(content="done")])
    response = asyncio.run(provider.complete(ModelRequest((), ())))

    assert response.content == "done"
    assert len(provider.requests) == 1
