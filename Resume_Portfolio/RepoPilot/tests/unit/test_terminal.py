from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

from repopilot.terminal import ConsoleTerminal, _completion_candidates, _redact_history_entry


def test_console_terminal_joins_backslash_continued_prompt_lines() -> None:
    responses = iter(("first line\\", "second line"))
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    terminal = ConsoleTerminal(read=read)

    assert terminal.read_prompt() == "first line\nsecond line"
    assert prompts == ["> ", "... "]


def test_console_terminal_strips_a_single_line_prompt() -> None:
    values: Iterator[str] = iter(("  inspect this  ",))
    terminal = ConsoleTerminal(read=lambda _prompt: next(values))

    assert terminal.read_prompt() == "inspect this"


def test_async_terminal_input_uses_prompt_async_without_a_nested_event_loop() -> None:
    prompts: list[str] = []

    class _PromptSession:
        async def prompt_async(self, prompt: str) -> str:
            prompts.append(prompt)
            return "  inspect this  "

    terminal = ConsoleTerminal()
    terminal._prompt_session = _PromptSession()  # noqa: SLF001 - verifies the adapter boundary.

    assert asyncio.run(terminal.read_prompt_async()) == "inspect this"
    assert prompts == ["> "]


def test_completion_candidates_support_slash_commands_and_bounded_workspace_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "src" / "models").mkdir()

    assert _completion_candidates(
        "/st", commands=("/status", "/stats", "/exit"), workspace=tmp_path
    ) == ("/status", "/stats")
    assert _completion_candidates("inspect src/m", commands=(), workspace=tmp_path) == (
        "src/main.py",
        "src/models/",
    )
    assert _completion_candidates("inspect src/", commands=(), workspace=tmp_path) == (
        "src/main.py",
        "src/models/",
    )
    assert _completion_candidates("/verify t", commands=("/verify",), workspace=None) == (
        "test",
        "typecheck",
    )


def test_path_completion_never_escapes_the_workspace(tmp_path: Path) -> None:
    assert _completion_candidates("inspect ../", commands=(), workspace=tmp_path) == ()


def test_terminal_history_redacts_common_api_key_forms() -> None:
    redacted = _redact_history_entry("DEEPSEEK_API_KEY=sk-abcdefghijklmnopqrst")

    assert "sk-abcdefghijklmnopqrst" not in redacted
    assert "[REDACTED" in redacted
