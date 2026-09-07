"""Durable, human-readable plans and todos for one interactive session."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from repopilot.core.contracts import JSONValue


class TodoStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class TodoItem:
    content: str
    status: TodoStatus = TodoStatus.PENDING
    active_form: str = ""

    def __post_init__(self) -> None:
        if not 1 <= len(self.content.strip()) <= 240:
            raise ValueError("todo content must contain 1-240 characters")
        if self.active_form and len(self.active_form.strip()) > 240:
            raise ValueError("todo active_form must contain at most 240 characters")

    def to_dict(self) -> dict[str, str]:
        return {
            "content": self.content.strip(),
            "status": self.status.value,
            "active_form": self.active_form.strip() or self.content.strip(),
        }


@dataclass(frozen=True, slots=True)
class SessionPlan:
    """A bounded plan that remains editable and inspectable after compaction."""

    summary: str
    approved: bool
    todos: tuple[TodoItem, ...]

    def __post_init__(self) -> None:
        if len(self.summary.strip()) > 500:
            raise ValueError("plan summary must contain at most 500 characters")
        if len(self.todos) > 20:
            raise ValueError("a session plan may contain at most 20 todos")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "summary": self.summary.strip(),
            "approved": self.approved,
            "todos": [item.to_dict() for item in self.todos],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, JSONValue]) -> SessionPlan:
        raw_todos = raw.get("todos", [])
        if not isinstance(raw_todos, list):
            raise ValueError("session plan todos must be a list")
        todos: list[TodoItem] = []
        for item in raw_todos:
            if not isinstance(item, dict):
                raise ValueError("session plan todo must be an object")
            status = item.get("status", TodoStatus.PENDING.value)
            try:
                parsed_status = TodoStatus(str(status))
            except ValueError as error:
                raise ValueError("session plan todo has invalid status") from error
            content = item.get("content")
            active_form = item.get("active_form", "")
            if not isinstance(content, str) or not isinstance(active_form, str):
                raise ValueError("session plan todo fields must be strings")
            todos.append(TodoItem(content, parsed_status, active_form))
        summary = raw.get("summary", "")
        approved = raw.get("approved", False)
        if not isinstance(summary, str) or not isinstance(approved, bool):
            raise ValueError("session plan summary and approval must be valid")
        return cls(summary, approved, tuple(todos))

    @classmethod
    def from_tool_items(
        cls, raw: list[dict[str, JSONValue]], *, previous: SessionPlan
    ) -> SessionPlan:
        """Validate a model TodoWrite payload while retaining human approval state."""

        todos: list[TodoItem] = []
        for item in raw:
            content = item.get("content")
            status = item.get("status")
            active_form = item.get("active_form")
            if (
                not isinstance(content, str)
                or not isinstance(status, str)
                or not isinstance(active_form, str)
            ):
                raise ValueError("each todo requires string content, status, and active_form")
            todos.append(TodoItem(content, TodoStatus(status), active_form))
        return cls(previous.summary or "Agent-managed task list", previous.approved, tuple(todos))

    def render_for_context(self) -> list[str]:
        return [f"[{item.status.value}] {item.content}" for item in self.todos]
