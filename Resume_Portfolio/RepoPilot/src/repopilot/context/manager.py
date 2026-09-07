"""Deterministic transcript projection and compaction for interactive sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from repopilot.core.contracts import Message


@dataclass(frozen=True, slots=True)
class ContextUsage:
    """Inspectable accounting for the compact context sent to a model."""

    source_messages: int
    retained_messages: int
    source_characters: int
    retained_characters: int
    pruned_tool_messages: int
    summary_characters: int


@dataclass(frozen=True, slots=True)
class ContextProjection:
    """The durable summary plus recent, bounded messages used for one model call."""

    messages: tuple[Message, ...]
    summary: str | None
    usage: ContextUsage


class ContextManager:
    """Keep a complete transcript on disk without repeatedly sending it all to a model."""

    def __init__(
        self,
        *,
        max_history_messages: int = 16,
        max_context_characters: int = 30_000,
        max_message_characters: int = 8_000,
        max_tool_output_characters: int = 4_000,
        max_summary_characters: int = 12_000,
    ) -> None:
        if (
            min(
                max_history_messages,
                max_context_characters,
                max_message_characters,
                max_tool_output_characters,
                max_summary_characters,
            )
            <= 0
        ):
            raise ValueError("context limits must be positive")
        self.max_history_messages = max_history_messages
        self.max_context_characters = max_context_characters
        self.max_message_characters = max_message_characters
        self.max_tool_output_characters = max_tool_output_characters
        self.max_summary_characters = max_summary_characters

    def project(
        self, messages: list[Message] | tuple[Message, ...], *, summary: str | None = None
    ) -> ContextProjection:
        """Return a recency-preserving bounded projection of the durable transcript."""
        source = tuple(messages)
        continuation_call_ids = self._latest_continuation_call_ids(source)
        retained_reversed: list[Message] = []
        consumed = 0
        pruned_tools = 0
        for message in reversed(source):
            required_continuation = (
                message.role == "assistant"
                and any(call.call_id in continuation_call_ids for call in message.tool_calls)
            ) or (
                message.role == "tool"
                and message.tool_call_id is not None
                and message.tool_call_id in continuation_call_ids
            )
            if len(retained_reversed) >= self.max_history_messages and not required_continuation:
                if message.role == "tool":
                    pruned_tools += 1
                continue
            limit = (
                self.max_tool_output_characters
                if message.role == "tool"
                else self.max_message_characters
            )
            content = (
                message.content if required_continuation else self._truncate(message.content, limit)
            )
            if content != message.content and message.role == "tool":
                pruned_tools += 1
            message_characters = len(content) + len(message.reasoning_content or "")
            available = self.max_context_characters - consumed
            if available <= 0 and not required_continuation:
                if message.role == "tool":
                    pruned_tools += 1
                continue
            if len(content) > available and not required_continuation:
                content = self._truncate(content, available)
                if message.role == "tool":
                    pruned_tools += 1
            retained_reversed.append(
                Message(
                    message.role,
                    content,
                    message.name,
                    message.tool_call_id,
                    message.tool_calls,
                    message.reasoning_content,
                )
            )
            consumed += message_characters
        retained, dropped_tools = self._retain_complete_tool_turns(
            tuple(reversed(retained_reversed))
        )
        pruned_tools += dropped_tools
        consumed = sum(
            len(message.content) + len(message.reasoning_content or "") for message in retained
        )
        normalized_summary = self._normalize_summary(summary)
        return ContextProjection(
            retained,
            normalized_summary,
            ContextUsage(
                source_messages=len(source),
                retained_messages=len(retained),
                source_characters=sum(
                    len(message.content) + len(message.reasoning_content or "")
                    for message in source
                ),
                retained_characters=consumed,
                pruned_tool_messages=pruned_tools,
                summary_characters=len(normalized_summary or ""),
            ),
        )

    def compact(
        self,
        messages: list[Message] | tuple[Message, ...],
        *,
        previous_summary: str | None = None,
    ) -> str:
        """Create a deterministic handoff summary without trusting tool-output text.

        This initial compactor intentionally does not call a model. It preserves a
        concise evidence trail even when a model is unavailable or a session crashes.
        """
        user_requests = [message.content for message in messages if message.role == "user"][-6:]
        assistant_answers = [
            message.content for message in messages if message.role == "assistant"
        ][-6:]
        observations = [
            self._tool_observation(message) for message in messages if message.role == "tool"
        ]
        sections: list[tuple[str, list[str]]] = []
        if previous_summary:
            sections.append(("Earlier summary", [self._normalize_summary(previous_summary) or ""]))
        if user_requests:
            sections.append(
                ("Recent user goals", [self._excerpt(item, 700) for item in user_requests])
            )
        if assistant_answers:
            sections.append(
                (
                    "Recent assistant conclusions",
                    [self._excerpt(item, 700) for item in assistant_answers],
                )
            )
        if observations:
            sections.append(("Tool evidence", observations[-12:]))
        if not sections:
            return "No prior session activity."
        rendered = ["Session handoff summary (generated deterministically):"]
        for heading, entries in sections:
            rendered.append(f"\n{heading}:")
            rendered.extend(f"- {entry}" for entry in entries if entry)
        return self._normalize_summary("\n".join(rendered)) or "No prior session activity."

    @staticmethod
    def _latest_continuation_call_ids(messages: tuple[Message, ...]) -> set[str]:
        """Keep the newest opaque-reasoning tool exchange intact for a follow-up call."""
        result_ids = {
            message.tool_call_id
            for message in messages
            if message.role == "tool" and message.tool_call_id is not None
        }
        for message in reversed(messages):
            if (
                message.role == "assistant"
                and message.reasoning_content is not None
                and message.tool_calls
                and all(call.call_id in result_ids for call in message.tool_calls)
            ):
                return {call.call_id for call in message.tool_calls}
        return set()

    @staticmethod
    def _retain_complete_tool_turns(
        messages: tuple[Message, ...],
    ) -> tuple[tuple[Message, ...], int]:
        """Drop incomplete native tool exchanges instead of emitting invalid history.

        OpenAI-compatible providers require every retained assistant tool-call
        message to be followed by its tool result(s). History clipping can split
        that exchange; omitting the partial exchange is safer than presenting a
        tool result as a user instruction or creating an invalid provider payload.
        """
        resolved_ids = {
            message.tool_call_id
            for message in messages
            if message.role == "tool" and message.tool_call_id is not None
        }
        complete_call_ids: set[str] = set()
        for message in messages:
            if message.role != "assistant" or not message.tool_calls:
                continue
            if all(call.call_id in resolved_ids for call in message.tool_calls):
                complete_call_ids.update(call.call_id for call in message.tool_calls)
        retained: list[Message] = []
        dropped_tools = 0
        for message in messages:
            if (
                message.role == "assistant"
                and message.tool_calls
                and any(call.call_id not in complete_call_ids for call in message.tool_calls)
            ):
                continue
            if (
                message.role == "tool"
                and message.tool_call_id is not None
                and message.tool_call_id not in complete_call_ids
            ):
                dropped_tools += 1
                continue
            retained.append(message)
        return tuple(retained), dropped_tools

    def _normalize_summary(self, summary: str | None) -> str | None:
        if not summary:
            return None
        return self._truncate(summary.strip(), self.max_summary_characters)

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        if limit <= 0:
            return ""
        marker = "\n...[context truncated by RepoPilot]"
        if limit <= len(marker):
            return value[:limit]
        return value[: limit - len(marker)] + marker

    @staticmethod
    def _excerpt(value: str, limit: int) -> str:
        return " ".join(ContextManager._truncate(value, limit).split())

    @staticmethod
    def _tool_observation(message: Message) -> str:
        try:
            raw = json.loads(message.content)
        except json.JSONDecodeError:
            return f"{message.name or 'tool'}: unparseable observation"
        if not isinstance(raw, dict):
            return f"{message.name or 'tool'}: non-object observation"
        name = str(raw.get("tool_name") or message.name or "tool")
        if bool(raw.get("ok")):
            return f"{name}: succeeded"
        error = ContextManager._excerpt(str(raw.get("error") or "unknown error"), 300)
        return f"{name}: failed ({error})"
