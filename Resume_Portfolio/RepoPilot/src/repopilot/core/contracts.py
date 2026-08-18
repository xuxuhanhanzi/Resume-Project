"""Typed contracts shared by models, tools, policy, tracing, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypeAlias, cast

# JSON flows across provider and protocol boundaries and is validated before it reaches
# typed domain contracts. Keeping this boundary explicitly dynamic avoids recursive
# container-invariance noise while strict types remain in the runtime itself.
JSONValue: TypeAlias = Any
MessageRole: TypeAlias = Literal["system", "user", "assistant", "tool"]


class RunStatus(StrEnum):
    """Durable top-level states for one agent run."""

    CREATED = "created"
    PREPARING = "preparing"
    RUNNING = "running"
    VERIFYING = "verifying"
    APPROVAL_REQUIRED = "approval_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorType(StrEnum):
    """Errors that drive retry and recovery decisions."""

    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    PERMISSION = "permission"
    CONFLICT = "conflict"
    EXECUTION = "execution"
    SECURITY = "security"
    INTERNAL = "internal"


class Permission(StrEnum):
    """Minimum permission required by a tool."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    HIGH_RISK = "high_risk"


class PolicyOutcome(StrEnum):
    """A structural policy decision made before execution."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True, slots=True)
class Message:
    """One provider-neutral conversation message."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        data: dict[str, JSONValue] = {"role": self.role, "content": self.content}
        if self.name is not None:
            data["name"] = self.name
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, JSONValue]) -> Message:
        return cls(
            role=cast(MessageRole, str(data["role"])),
            content=str(data.get("content", "")),
            name=str(data["name"]) if data.get("name") is not None else None,
            tool_call_id=(
                str(data["tool_call_id"]) if data.get("tool_call_id") is not None else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A model-visible tool schema plus runtime-only safety metadata."""

    name: str
    description: str
    input_schema: dict[str, JSONValue]
    permission: Permission = Permission.READ
    timeout_seconds: float = 10.0
    read_only: bool = True
    idempotent: bool = True

    def provider_schema(self) -> dict[str, JSONValue]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A validated provider-neutral request to execute one tool."""

    call_id: str
    name: str
    arguments: dict[str, JSONValue]

    def to_dict(self) -> dict[str, JSONValue]:
        return {"call_id": self.call_id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict[str, JSONValue]) -> ToolCall:
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be an object")
        return cls(str(data["call_id"]), str(data["name"]), arguments)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A structured observation returned to the agent after a tool call."""

    call_id: str
    tool_name: str
    ok: bool
    data: JSONValue = None
    error: str | None = None
    error_type: ErrorType | None = None
    recoverable: bool = False
    side_effect: bool = False
    cached: bool = False

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "error_type": self.error_type.value if self.error_type is not None else None,
            "recoverable": self.recoverable,
            "side_effect": self.side_effect,
            "cached": self.cached,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSONValue]) -> ToolResult:
        raw_error_type = data.get("error_type")
        return cls(
            call_id=str(data["call_id"]),
            tool_name=str(data["tool_name"]),
            ok=bool(data["ok"]),
            data=data.get("data"),
            error=str(data["error"]) if data.get("error") is not None else None,
            error_type=ErrorType(str(raw_error_type)) if raw_error_type is not None else None,
            recoverable=bool(data.get("recoverable", False)),
            side_effect=bool(data.get("side_effect", False)),
            cached=bool(data.get("cached", False)),
        )


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Token accounting reported by a model backend."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Provider-neutral model request."""

    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    temperature: float = 0.0
    max_output_tokens: int = 1024


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Provider-neutral model response."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: ModelUsage = field(default_factory=ModelUsage)
    model: str = "unknown"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """The decision returned by the permission layer."""

    outcome: PolicyOutcome
    reason: str


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Deterministic evidence used to accept or reject a proposed finish."""

    passed: bool
    summary: str
    recoverable: bool = False
    details: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(slots=True)
class AgentState:
    """Serializable state required to continue an interrupted run."""

    run_id: str
    task_id: str
    status: RunStatus = RunStatus.CREATED
    iteration: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    messages: list[Message] = field(default_factory=list)
    pending_calls: list[ToolCall] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    final_answer: str = ""
    failure_reason: str = ""

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "iteration": self.iteration,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "messages": [message.to_dict() for message in self.messages],
            "pending_calls": [call.to_dict() for call in self.pending_calls],
            "plan": list(self.plan),
            "completed_steps": list(self.completed_steps),
            "final_answer": self.final_answer,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSONValue]) -> AgentState:
        raw_messages = data.get("messages", [])
        messages = [
            Message.from_dict(cast(dict[str, JSONValue], item))
            for item in raw_messages
            if isinstance(item, dict)
        ]
        raw_plan = data.get("plan", [])
        raw_completed = data.get("completed_steps", [])
        raw_pending = data.get("pending_calls", [])
        return cls(
            run_id=str(data["run_id"]),
            task_id=str(data["task_id"]),
            status=RunStatus(str(data.get("status", RunStatus.CREATED.value))),
            iteration=int(cast(int, data.get("iteration", 0))),
            tool_calls=int(cast(int, data.get("tool_calls", 0))),
            input_tokens=int(cast(int, data.get("input_tokens", 0))),
            output_tokens=int(cast(int, data.get("output_tokens", 0))),
            messages=messages,
            pending_calls=[
                ToolCall.from_dict(cast(dict[str, JSONValue], item))
                for item in raw_pending
                if isinstance(item, dict)
            ],
            plan=[str(item) for item in raw_plan] if isinstance(raw_plan, list) else [],
            completed_steps=(
                [str(item) for item in raw_completed] if isinstance(raw_completed, list) else []
            ),
            final_answer=str(data.get("final_answer", "")),
            failure_reason=str(data.get("failure_reason", "")),
        )
