"""Small, explicit guard for user requests that require a named structured tool."""

from __future__ import annotations

from collections.abc import Iterable

_REQUIREMENT_CUES = (
    "must call",
    "must use",
    "call the tool",
    "use the tool",
    "必须调用",
    "必须使用",
    "请调用",
    "请使用",
    "调用工具",
    "使用工具",
)


def required_tool_from_prompt(prompt: str, names: Iterable[str]) -> str | None:
    """Return one explicitly named tool the user required, otherwise ``None``.

    This deliberately does not guess from ordinary natural language.  It only
    activates when a real registered tool name and an imperative cue co-occur,
    avoiding accidental forced calls such as a discussion *about* ``run_tests``.
    """

    normalized = prompt.casefold()
    if not any(cue in normalized for cue in _REQUIREMENT_CUES):
        return None
    candidates = sorted({name for name in names}, key=len, reverse=True)
    for name in candidates:
        if name.casefold() in normalized:
            return name
    return None
