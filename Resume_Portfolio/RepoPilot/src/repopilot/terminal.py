"""Portable interactive terminal primitives with a safe dependency-free fallback."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from repopilot.security.redaction import redact_text


def _redact_history_entry(value: str) -> str:
    """Keep terminal history useful without persisting common credential forms."""
    return redact_text(value)


def _completion_candidates(
    text_before_cursor: str,
    *,
    commands: Iterable[str],
    workspace: Path | None,
    max_paths: int = 100,
) -> tuple[str, ...]:
    """Return bounded slash-command or workspace-path completions.

    Filesystem completion deliberately inspects only the requested parent
    directory. It never recursively scans a repository and never suggests a
    path outside the configured workspace.
    """
    stripped = text_before_cursor.lstrip()
    if stripped.startswith("/") and not any(character.isspace() for character in stripped):
        return tuple(command for command in commands if command.startswith(stripped))
    if stripped.startswith("/"):
        words = stripped.split()
        if len(words) == 2:
            return tuple(
                candidate
                for candidate in _SLASH_ARGUMENTS.get(words[0], ())
                if candidate.startswith(words[1])
            )

    words = text_before_cursor.split()
    if not words or workspace is None:
        return ()
    token = words[-1]
    if not (token.startswith(".") or "/" in token or "\\" in token):
        return ()
    try:
        root = workspace.resolve()
        candidate = (root / token).resolve(strict=False)
        parent = candidate if token.endswith(("/", "\\")) else candidate.parent
        parent.relative_to(root)
    except (OSError, ValueError):
        return ()
    if not parent.is_dir():
        return ()
    prefix = "" if token.endswith(("/", "\\")) else candidate.name.casefold()
    matches: list[str] = []
    try:
        entries = sorted(parent.iterdir(), key=lambda item: item.name.casefold())
    except OSError:
        return ()
    for entry in entries:
        if len(matches) >= max_paths or not entry.name.casefold().startswith(prefix):
            continue
        try:
            relative = entry.resolve(strict=False).relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        matches.append(relative + "/" if entry.is_dir() else relative)
    return tuple(matches)


_SLASH_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "/plan": ("draft", "approve", "clear"),
    "/rewind": ("list",),
    "/todo": ("add", "start", "done"),
    "/verify": ("list", "last", "all", "test", "lint", "typecheck", "build"),
}


class ConsoleTerminal:
    """Read prompts with optional persistent history and Tab completion.

    Prompt Toolkit provides consistent Up/Down history and Tab completion on
    Windows, macOS and Linux. If it is unavailable, or standard input/output
    are not real terminals, RepoPilot preserves its original ``input`` based
    behaviour. A trailing backslash continues a natural-language prompt on a
    new line in both modes.
    """

    def __init__(
        self,
        *,
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
    ) -> None:
        self._read = read
        self._write = write
        self._prompt_session: Any | None = None
        self._advanced_input = False

    @property
    def advanced_input(self) -> bool:
        """Whether RepoPilot-managed history and completion are active."""
        return self._advanced_input

    def configure_interactive(
        self,
        *,
        history_path: Path,
        workspace: Path,
        commands: Iterable[str],
    ) -> None:
        """Enable Prompt Toolkit only for an actual interactive console."""
        if self._read is not input or not (sys.stdin.isatty() and sys.stdout.isatty()):
            return
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import Completer, Completion
            from prompt_toolkit.history import FileHistory
        except ImportError:
            return

        history_path.parent.mkdir(parents=True, exist_ok=True)
        command_words = tuple(sorted(set(commands)))

        class RedactingFileHistory(FileHistory):
            def store_string(self, string: str) -> None:
                super().store_string(_redact_history_entry(string))

        class RepoPilotCompleter(Completer):
            def get_completions(self, document: Any, complete_event: Any) -> Iterable[Any]:
                del complete_event
                before_cursor = document.text_before_cursor
                candidates = _completion_candidates(
                    before_cursor,
                    commands=command_words,
                    workspace=workspace,
                )
                if not candidates:
                    return ()
                words = before_cursor.split()
                current_word = words[-1] if words else before_cursor
                return tuple(
                    Completion(candidate, start_position=-len(current_word))
                    for candidate in candidates
                )

        self._prompt_session = PromptSession(
            history=RedactingFileHistory(str(history_path)),
            completer=RepoPilotCompleter(),
            complete_while_typing=False,
            enable_history_search=True,
        )
        self._advanced_input = True

    def read_prompt(self, prompt: str = "> ") -> str:
        """Read a prompt synchronously for embedding callers and unit tests."""
        lines: list[str] = []
        current_prompt = prompt
        while True:
            if self._prompt_session is not None:
                raise RuntimeError("advanced terminal input requires read_prompt_async()")
            line = self._read(current_prompt)
            if line.endswith("\\"):
                lines.append(line[:-1])
                current_prompt = "... "
                continue
            lines.append(line)
            return "\n".join(lines).strip()

    async def read_prompt_async(self, prompt: str = "> ") -> str:
        """Return one prompt without starting a nested asyncio event loop."""
        lines: list[str] = []
        current_prompt = prompt
        while True:
            line = (
                await self._prompt_session.prompt_async(current_prompt)
                if self._prompt_session is not None
                else self._read(current_prompt)
            )
            if line.endswith("\\"):
                lines.append(line[:-1])
                current_prompt = "... "
                continue
            lines.append(line)
            return "\n".join(lines).strip()

    def status(self, message: str) -> None:
        """Render a short live status line without terminal-control sequences."""
        self._write(message)
