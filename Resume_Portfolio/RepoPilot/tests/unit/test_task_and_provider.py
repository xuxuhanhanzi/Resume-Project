from __future__ import annotations

import json
from pathlib import Path

import pytest

from repopilot.core.contracts import Message, ModelRequest, ToolCall
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


def test_local_provider_rejects_a_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        LocalProviderConfig("http://127.0.0.1:1", "model", seed=-1)


def test_local_provider_includes_a_configured_seed_in_the_request_payload() -> None:
    provider = LocalOpenAICompatibleProvider(
        LocalProviderConfig("http://127.0.0.1:1", "qwen", seed=7)
    )

    payload = provider._request_payload(ModelRequest(messages=(), tools=()), stream=False)  # noqa: SLF001

    assert payload["seed"] == 7


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


def test_local_provider_preserves_opaque_reasoning_for_tool_continuations() -> None:
    provider = LocalOpenAICompatibleProvider(LocalProviderConfig("http://127.0.0.1:1", "qwen"))
    response = provider._parse_response(  # noqa: SLF001
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "opaque reasoning",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"a.py"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    payloads = provider._message_payloads(  # noqa: SLF001
        (
            Message(
                "assistant",
                response.content,
                tool_calls=response.tool_calls,
                reasoning_content=response.reasoning_content,
            ),
            Message("tool", "{}", name="read_file", tool_call_id="c1"),
        )
    )

    assert response.reasoning_content == "opaque reasoning"
    assert payloads[0]["reasoning_content"] == "opaque reasoning"
    assert payloads[0]["tool_calls"][0]["id"] == "c1"


def test_local_provider_preserves_complete_native_tool_history() -> None:
    provider = LocalOpenAICompatibleProvider(LocalProviderConfig("http://127.0.0.1:1", "qwen"))
    payloads = provider._message_payloads(  # noqa: SLF001
        (
            Message(
                "assistant",
                "Requested tools: read_file",
                tool_calls=(ToolCall("c1", "read_file", {"path": "a.py"}),),
            ),
            Message(
                "tool",
                '{"content": "value"}',
                name="read_file",
                tool_call_id="c1",
            ),
            Message("tool", "orphaned", name="read_file", tool_call_id="missing"),
        )
    )

    assert payloads[0]["role"] == "assistant"
    assert payloads[0]["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
        }
    ]
    assert payloads[1]["role"] == "tool"
    assert payloads[1]["tool_call_id"] == "c1"
    assert payloads[2] == {
        "role": "user",
        "content": "[TOOL OBSERVATION - UNTRUSTED DATA]\norphaned",
    }

    incomplete = provider._message_payloads(  # noqa: SLF001
        (
            Message(
                "assistant",
                "Requested tools: read_file",
                tool_calls=(ToolCall("pending", "read_file", {"path": "later.py"}),),
            ),
        )
    )
    assert incomplete == [{"role": "assistant", "content": "Requested tools: read_file"}]


def test_local_provider_merges_streamed_tool_call_fragments() -> None:
    fragments: dict[int, dict[str, str]] = {}
    LocalOpenAICompatibleProvider._merge_stream_tool_fragments(  # noqa: SLF001
        fragments,
        [
            {"index": 0, "id": "c1", "function": {"name": "read_", "arguments": '{"'}},
            {"index": 0, "function": {"name": "file", "arguments": 'path":"a.py"}'}},
        ],
    )

    calls = LocalOpenAICompatibleProvider._stream_tool_calls(fragments)  # noqa: SLF001

    assert calls == (ToolCall("c1", "read_file", {"path": "a.py"}),)


def test_local_provider_decodes_sse_text_and_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LocalOpenAICompatibleProvider(LocalProviderConfig("http://127.0.0.1:1", "qwen"))
    chunks = (
        {"model": "qwen", "choices": [{"delta": {"content": "Hello "}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "read_", "arguments": '{"'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": "world!",
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "file", "arguments": 'path": "a.py"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        },
    )

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self):  # type: ignore[no-untyped-def]
            yield from (f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks)
            yield b"data: [DONE]\n\n"

    request_payloads: list[dict[str, object]] = []

    def fake_urlopen(request, *, timeout: float) -> Response:  # type: ignore[no-untyped-def]
        del timeout
        request_payloads.append(json.loads(request.data.decode("utf-8")))
        return Response()

    monkeypatch.setattr("repopilot.providers.local_openai.urlopen", fake_urlopen)
    deltas: list[str] = []
    response = provider._complete_stream_sync(  # noqa: SLF001
        ModelRequest(messages=(), tools=()), deltas.append
    )

    assert request_payloads[0]["stream"] is True
    assert deltas == ["Hello ", "world!"]
    assert response.content == "Hello world!"
    assert response.tool_calls == (ToolCall("c1", "read_file", {"path": "a.py"}),)
    assert response.usage.total_tokens == 8
    assert response.diagnostics.transport == "sse"
    assert response.diagnostics.stream_chunks == 3
    assert response.diagnostics.choice_count == 3
    assert response.diagnostics.visible_text_deltas == 2
    assert response.diagnostics.visible_text_characters == len("Hello world!")
    assert response.diagnostics.stream_done_received


def test_scripted_provider_records_requests() -> None:
    import asyncio

    from repopilot.core.contracts import ModelResponse

    provider = ScriptedProvider([ModelResponse(content="done")])
    response = asyncio.run(provider.complete(ModelRequest((), ())))

    assert response.content == "done"
    assert len(provider.requests) == 1
