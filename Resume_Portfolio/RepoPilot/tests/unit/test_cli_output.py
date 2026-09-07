from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from repopilot.cli import (
    _interactive_welcome,
    _preflight_command,
    _prompt_with_preflight_evidence,
    _render_edit_preview,
    _render_event,
    _render_turn_result,
    _safe_terminal_text,
    _TestOnlyApprovalHandler,
    build_parser,
)
from repopilot.core.contracts import AgentState, ToolCall
from repopilot.core.events import RuntimeEvent, RuntimeEventKind
from repopilot.runtime.policy import PermissionMode
from repopilot.session.models import SessionMetadata
from repopilot.session.runtime import SessionTurnResult
from repopilot.session.workflow import TurnWorkflowReport, WorkflowCheck


def test_jsonl_output_is_a_single_machine_readable_event(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _render_event(
        RuntimeEvent(RuntimeEventKind.TOOL_CALL_STARTED, {"call_id": "c1", "tool": "read_file"}),
        output_format="jsonl",
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "type": "event",
        "event": {
            "kind": "tool_call_started",
            "data": {"call_id": "c1", "tool": "read_file"},
        },
    }


def test_cli_exposes_the_jsonl_output_contract() -> None:
    args = build_parser().parse_args(["--output-format", "jsonl", "--trust", "-p", "status"])
    assert args.output_format == "jsonl"


def test_interactive_text_streaming_is_default_and_can_be_disabled() -> None:
    default_args = build_parser().parse_args([])
    buffered_args = build_parser().parse_args(["--no-stream"])

    assert default_args.stream is True
    assert buffered_args.stream is False


def test_allow_tests_is_a_narrow_non_interactive_approval_mode() -> None:
    args = build_parser().parse_args(["--trust", "--allow-tests", "-p", "status"])
    handler = _TestOnlyApprovalHandler()

    assert args.allow_tests is True
    assert asyncio.run(
        handler.approve(ToolCall(call_id="test", name="run_tests", arguments={}), "test")
    )
    assert not asyncio.run(
        handler.approve(ToolCall(call_id="shell", name="run_shell", arguments={}), "shell")
    )


def test_cli_exposes_preflight_test_option() -> None:
    args = build_parser().parse_args(["--trust", "--preflight-tests"])

    assert args.preflight_tests is True


def test_preflight_evidence_is_attached_once_to_a_natural_language_prompt() -> None:
    prompt = _prompt_with_preflight_evidence("检查测试", "Exit code: 0")

    assert prompt.startswith("检查测试")
    assert "<preflight-test-report>" in prompt
    assert _prompt_with_preflight_evidence("继续分析", None) == "继续分析"


def test_preflight_pytest_command_ignores_console_interrupts_in_its_child() -> None:
    command = _preflight_command(("python", "-m", "pytest"))

    assert command[:2] == ("python", "-c")
    assert "SIG_IGN" in command[2]


def test_interactive_welcome_uses_repopilot_identity_and_session_details() -> None:
    metadata = SessionMetadata.create(
        session_id="12345678-abcd-efgh-ijkl-123456789abc",
        project_root=Path("C:/project"),
        model="deepseek-v4-flash",
        permission_mode=PermissionMode.MANUAL,
    )

    welcome = _interactive_welcome(metadata)

    assert "RepoPilot" in welcome
    assert "Project:" in welcome
    assert "deepseek-v4-flash" in welcome
    assert "12345678" in welcome
    assert "/model" in welcome
    assert "Ctrl+C" in welcome


def test_edit_preview_renders_a_unified_diff_without_terminal_control_sequences(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _render_edit_preview(
        ToolCall(
            "edit",
            "apply_patch",
            {"path": "module.py", "old_text": "one\n", "new_text": "two\n"},
        )
    )

    output = capsys.readouterr().out
    assert "--- a/module.py" in output
    assert "+two" in output
    assert _safe_terminal_text("a\x1b[2Jb") == "a^[[2Jb"


def test_text_output_suppresses_runtime_status_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _render_event(
        RuntimeEvent(RuntimeEventKind.TOOL_CALL_STARTED, {"call_id": "c1", "tool": "read_file"}),
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_stream_mode_prints_only_model_text_not_tool_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _render_event(
        RuntimeEvent(RuntimeEventKind.MODEL_OUTPUT_DELTA, {"text": "streamed text"}),
        stream=True,
    )
    _render_event(
        RuntimeEvent(RuntimeEventKind.TOOL_CALL_STARTED, {"call_id": "c1", "tool": "read_file"}),
        stream=True,
    )

    assert capsys.readouterr().out == "streamed text"


def test_text_output_shows_one_human_readable_start_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _render_event(
        RuntimeEvent(RuntimeEventKind.SESSION_INITIALIZING, {"project_root": "C:/project"}),
    )

    assert capsys.readouterr().out == "正在分析项目，请稍候…\n"


def test_session_initialization_is_immediately_visible_in_jsonl(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _render_event(
        RuntimeEvent(RuntimeEventKind.SESSION_INITIALIZING, {"project_root": "C:/project"}),
        output_format="jsonl",
    )

    assert json.loads(capsys.readouterr().out) == {
        "type": "event",
        "event": {
            "kind": "session_initializing",
            "data": {"project_root": "C:/project"},
        },
    }


def test_final_turn_output_includes_only_runtime_backed_workflow_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = SessionMetadata.create(
        session_id="12345678-abcd-efgh-ijkl-123456789abc",
        project_root=Path("C:/project"),
        model="scripted",
        permission_mode=PermissionMode.MANUAL,
    )
    result = SessionTurnResult(
        metadata,
        "Updated the module.",
        AgentState(metadata.session_id, "session"),
        TurnWorkflowReport(
            ("module.py",),
            (WorkflowCheck("run_tests", ("python", "-m", "pytest"), True),),
        ),
    )

    _render_turn_result(result, output_format="text")

    output = capsys.readouterr().out
    assert "Updated the module." in output
    assert "Changed:" in output
    assert "module.py" in output
    assert "passed: run_tests" in output
