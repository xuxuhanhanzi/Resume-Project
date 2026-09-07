"""Declarative, non-executable project hooks for auditable session guidance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from repopilot.core.events import RuntimeEvent, RuntimeEventKind

_EVENTS = {
    "session_start",
    "user_prompt_submit",
    "pre_tool_use",
    "post_tool_use",
    "tool_failure",
    "verification_completed",
    "session_stop",
}


@dataclass(frozen=True, slots=True)
class HookNotice:
    """A project-authored, visible reminder; it cannot execute or modify anything."""

    event: str
    message: str
    tools: tuple[str, ...] = ()
    matcher: str = ""


class HookDispatcher:
    """Load a bounded JSON declaration and emit notices for matching runtime events.

    Project hooks deliberately have no command, Python, or HTTP execution surface.
    Any future executable hook must be separately designed to pass through the
    ordinary policy, approval, runner, and audit layers.
    """

    def __init__(self, notices: tuple[HookNotice, ...] = ()) -> None:
        self.notices = notices

    @classmethod
    def load(cls, project_root: Path) -> HookDispatcher:
        path = project_root.resolve() / ".repopilot" / "hooks.json"
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid project hook configuration: {error}") from error
        hooks = raw.get("hooks", []) if isinstance(raw, dict) else []
        if not isinstance(hooks, list) or len(hooks) > 32:
            raise ValueError("project hooks must contain at most 32 entries")
        notices: list[HookNotice] = []
        for item in hooks:
            if not isinstance(item, dict):
                raise ValueError("each project hook must be an object")
            event = item.get("event")
            message = item.get("message")
            raw_tools = item.get("tools", [])
            matcher = item.get("matcher", "")
            if (
                not isinstance(event, str)
                or event not in _EVENTS
                or not isinstance(message, str)
                or not 1 <= len(message.strip()) <= 500
                or not isinstance(raw_tools, list)
                or not all(isinstance(tool, str) and tool for tool in raw_tools)
                or not isinstance(matcher, str)
                or len(matcher) > 100
            ):
                raise ValueError(
                    "project hook has an invalid event, message, matcher, or tool filter"
                )
            notices.append(HookNotice(event, message.strip(), tuple(raw_tools), matcher))
        return cls(tuple(notices))

    def dispatch(self, event: RuntimeEvent) -> tuple[HookNotice, ...]:
        """Return notices for a kernel event without executing project-provided code."""
        hook_event = self._hook_event(event)
        if hook_event is None:
            return ()
        tool = event.data.get("tool")
        return self.notices_for(hook_event, tool=tool if isinstance(tool, str) else None)

    def notices_for(self, event: str, *, tool: str | None = None) -> tuple[HookNotice, ...]:
        """Return lifecycle or tool notices without executing project-provided code."""
        return tuple(
            notice
            for notice in self.notices
            if notice.event == event
            and (not notice.tools or (tool is not None and tool in notice.tools))
            and (not notice.matcher or (tool is not None and fnmatchcase(tool, notice.matcher)))
        )

    @staticmethod
    def _hook_event(event: RuntimeEvent) -> str | None:
        if event.kind is RuntimeEventKind.TOOL_CALL_PROPOSED:
            return "pre_tool_use"
        if event.kind is RuntimeEventKind.TOOL_CALL_COMPLETED:
            return "post_tool_use" if event.data.get("ok") else "tool_failure"
        if event.kind is RuntimeEventKind.VERIFICATION_COMPLETED:
            return "verification_completed"
        if event.kind is RuntimeEventKind.TURN_STARTED:
            return "user_prompt_submit"
        return None
