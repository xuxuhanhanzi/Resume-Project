"""Multi-Agent harness with explicit state graph and message envelopes (P11).

This module implements a minimal but real multi-agent coordination layer:

  * ``AgentMessage`` — inter-agent message envelope with task ID,
    correlation ID, sender/recipient, permission level, and budget.
  * ``AgentNode`` — a named agent with a role and provider.
  * ``StateGraph`` — explicit edges defining planner → executor →
    verifier → reviewer → finish/replan.
  * ``MultiAgentHarness`` — top-level coordinator that drives the graph,
    enforces write-serialisation, and aggregates traces.

Design constraints (from the handoff document):
  * Parallel execution is restricted to read-only (no side-effect) tools.
  * Write operations stay serial and require approval.
  * Drift detection: goal summary, plan deviation, evidence gaps, repeated
    tool calls.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import time
from contextlib import suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from repopilot.core.contracts import (
    AgentState,
    JSONValue,
    Message,
    ModelRequest,
    RunStatus,
)
from repopilot.orchestration.routing import RouteTarget, RoutingDecision, RoutingMode, TaskRouter
from repopilot.providers.base import ModelProvider
from repopilot.task import PublicTaskSpec
from repopilot.tools.coding import capture_text_snapshot


class AgentRole(StrEnum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    REVIEWER = "reviewer"


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """Inter-agent message envelope."""

    sender: str
    recipient: str
    task_id: str
    correlation_id: str
    content: str
    permission: str = "read"
    budget_remaining: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class AgentNode:
    """A named agent node in the state graph."""

    name: str
    role: AgentRole
    system_prompt: str = ""


class RuntimeExecutor(Protocol):
    """The existing single-agent runtime surface used by the executor node."""

    async def run(
        self, task: PublicTaskSpec, *, run_id: str | None = None, resume: bool = True
    ) -> AgentState: ...


@dataclass(frozen=True, slots=True)
class RouteBinding:
    """Concrete runtime components selected by one routing target."""

    executor: RuntimeExecutor
    planner_provider: ModelProvider
    reviewer_provider: ModelProvider


@dataclass(slots=True)
class MultiAgentRunState:
    """Durable coordinator state stored independently from executor checkpoints."""

    run_id: str
    task_id: str
    status: RunStatus = RunStatus.CREATED
    current_node: str = "planner"
    cycle: int = 0
    executor_run_id: str = ""
    plan: str = ""
    review_feedback: str = ""
    final_answer: str = ""
    failure_reason: str = ""
    routing_mode: str = ""
    route_target: str = ""
    routing_reason: str = ""
    routing_confidence: float = 0.0
    routing_cascade: list[str] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "current_node": self.current_node,
            "cycle": self.cycle,
            "executor_run_id": self.executor_run_id,
            "plan": self.plan,
            "review_feedback": self.review_feedback,
            "final_answer": self.final_answer,
            "failure_reason": self.failure_reason,
            "routing_mode": self.routing_mode,
            "route_target": self.route_target,
            "routing_reason": self.routing_reason,
            "routing_confidence": self.routing_confidence,
            "routing_cascade": list(self.routing_cascade),
            "messages": [
                {
                    "sender": message.sender,
                    "recipient": message.recipient,
                    "task_id": message.task_id,
                    "correlation_id": message.correlation_id,
                    "content": message.content,
                    "permission": message.permission,
                    "budget_remaining": message.budget_remaining,
                    "timestamp": message.timestamp,
                }
                for message in self.messages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, JSONValue]) -> MultiAgentRunState:
        raw_messages = data.get("messages", [])
        messages: list[AgentMessage] = []
        if isinstance(raw_messages, list):
            for raw in raw_messages:
                if not isinstance(raw, dict):
                    continue
                messages.append(
                    AgentMessage(
                        sender=str(raw["sender"]),
                        recipient=str(raw["recipient"]),
                        task_id=str(raw["task_id"]),
                        correlation_id=str(raw["correlation_id"]),
                        content=str(raw.get("content", "")),
                        permission=str(raw.get("permission", "read")),
                        budget_remaining=float(raw.get("budget_remaining", 0.0)),
                        timestamp=float(raw.get("timestamp", time.time())),
                    )
                )
        return cls(
            run_id=str(data["run_id"]),
            task_id=str(data["task_id"]),
            status=RunStatus(str(data.get("status", RunStatus.CREATED.value))),
            current_node=str(data.get("current_node", "planner")),
            cycle=int(cast(int, data.get("cycle", 0))),
            executor_run_id=str(data.get("executor_run_id", "")),
            plan=str(data.get("plan", "")),
            review_feedback=str(data.get("review_feedback", "")),
            final_answer=str(data.get("final_answer", "")),
            failure_reason=str(data.get("failure_reason", "")),
            routing_mode=str(data.get("routing_mode", "")),
            route_target=str(data.get("route_target", "")),
            routing_reason=str(data.get("routing_reason", "")),
            routing_confidence=float(data.get("routing_confidence", 0.0)),
            routing_cascade=[
                str(item) for item in cast(list[JSONValue], data.get("routing_cascade", []))
            ],
            messages=messages,
        )


class MultiAgentCheckpointStore:
    """Atomic checkpoint for coordinator transitions and message envelopes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, state: MultiAgentRunState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def load(self) -> MultiAgentRunState | None:
        if not self.path.exists():
            return None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("multi-agent checkpoint root must be an object")
        return MultiAgentRunState.from_dict(cast(dict[str, JSONValue], raw))


class StateGraph:
    """Explicit state graph for multi-agent coordination.

    Default topology:
        planner → executor → verifier → reviewer → finish
                                  ↑         |
                                  └─ replan ┘
    """

    def __init__(self) -> None:
        self._nodes: dict[str, AgentNode] = {}
        self._edges: dict[str, list[str]] = {}
        self._add_default_graph()

    def _add_default_graph(self) -> None:
        planner = AgentNode("planner", AgentRole.PLANNER, "Plan the task.")
        executor = AgentNode("executor", AgentRole.EXECUTOR, "Execute tools.")
        verifier = AgentNode("verifier", AgentRole.VERIFIER, "Verify results.")
        reviewer = AgentNode("reviewer", AgentRole.REVIEWER, "Review quality.")
        for node in (planner, executor, verifier, reviewer):
            self._nodes[node.name] = node
        self._edges = {
            "planner": ["executor"],
            "executor": ["verifier"],
            "verifier": ["reviewer", "executor"],
            "reviewer": ["finish", "executor"],
        }

    @property
    def nodes(self) -> dict[str, AgentNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._edges.items()}

    def next_nodes(self, current: str) -> list[str]:
        return list(self._edges.get(current, []))


class MultiAgentHarness:
    """Top-level coordinator driving a state graph of agents.

    Planner and reviewer are isolated read-only model calls. The executor is the
    existing ``AgentRuntime``, which retains sole ownership of tools, writes,
    deterministic verification and its idempotent execution journal. Coordinator
    checkpoints make node transitions resumable without replaying a completed
    executor run.
    """

    def __init__(
        self,
        *,
        correlation_id: str | None = None,
        artifacts_root: Path | None = None,
    ) -> None:
        self.graph = StateGraph()
        self.correlation_id = correlation_id or uuid4().hex[:12]
        self.artifacts_root = artifacts_root
        self.messages: list[AgentMessage] = []
        self.drift_flags: list[dict[str, Any]] = []

    def detect_drift(
        self,
        state: AgentState,
        *,
        repeated_tool_threshold: int = 3,
        evidence_gap: bool = False,
        plan_deviation: bool = False,
    ) -> list[str]:
        """Detect drift signals: repeated calls, evidence gaps, plan deviations."""
        flags: list[str] = []
        tool_counts: dict[str, int] = {}
        for msg in state.messages:
            if msg.role == "tool" and msg.name:
                tool_counts[msg.name] = tool_counts.get(msg.name, 0) + 1
        for tool, count in tool_counts.items():
            if count >= repeated_tool_threshold:
                flags.append(f"repeated_tool:{tool}:{count}")
                self.drift_flags.append({"type": "repeated_tool", "tool": tool, "count": count})
        if evidence_gap:
            flags.append("evidence_gap")
            self.drift_flags.append({"type": "evidence_gap"})
        if plan_deviation:
            flags.append("plan_deviation")
            self.drift_flags.append({"type": "plan_deviation"})
        return flags

    def send(
        self,
        sender: str,
        recipient: str,
        task_id: str,
        content: str,
        *,
        permission: str = "read",
        budget_remaining: float = 0.0,
    ) -> AgentMessage:
        msg = AgentMessage(
            sender=sender,
            recipient=recipient,
            task_id=task_id,
            correlation_id=self.correlation_id,
            content=content,
            permission=permission,
            budget_remaining=budget_remaining,
        )
        self.messages.append(msg)
        return msg

    def trace_summary(self) -> dict[str, Any]:
        """Aggregate trace: message count, drift flags, node transitions."""
        node_transitions: list[str] = []
        for msg in self.messages:
            node_transitions.append(f"{msg.sender}→{msg.recipient}")
        return {
            "correlation_id": self.correlation_id,
            "total_messages": len(self.messages),
            "drift_flags": self.drift_flags,
            "node_transitions": node_transitions,
            "graph_topology": self.graph.edges,
        }

    async def run(
        self,
        task: PublicTaskSpec,
        *,
        executor: RuntimeExecutor,
        planner_provider: ModelProvider,
        reviewer_provider: ModelProvider,
        run_id: str | None = None,
        resume: bool = True,
        max_cycles: int = 2,
        timeout_seconds: float | None = None,
        cancel_event: asyncio.Event | None = None,
        router: TaskRouter | None = None,
        routing_provider: ModelProvider | None = None,
        route_bindings: dict[RouteTarget, RouteBinding] | None = None,
    ) -> MultiAgentRunState:
        """Execute planner → runtime/verifier → reviewer with durable transitions.

        A reviewer rejection starts another executor cycle with the feedback added
        to the public problem statement. Planner and reviewer receive no tools, so
        the executor remains the single writer. Process cancellation preserves a
        RUNNING checkpoint and can be resumed with the same ``run_id``.
        """
        if self.artifacts_root is None:
            raise ValueError("artifacts_root is required for executable orchestration")
        if max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        actual_run_id = run_id or f"multi_{uuid4().hex[:12]}"
        run_dir = self.artifacts_root / actual_run_id
        checkpoint = MultiAgentCheckpointStore(run_dir / "checkpoint.json")
        state = checkpoint.load() if resume else None
        if state is not None and (state.run_id != actual_run_id or state.task_id != task.task_id):
            raise ValueError("multi-agent checkpoint identity mismatch")
        if state is None:
            state = MultiAgentRunState(actual_run_id, task.task_id, status=RunStatus.RUNNING)
            baseline = capture_text_snapshot(task.workspace, include_paths=task.allowed_paths)
            self._save_baseline(run_dir, baseline)
            checkpoint.save(state)
        else:
            baseline = self._load_baseline(run_dir)
            if state.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                self.messages = list(state.messages)
                return state
            self.correlation_id = (
                state.messages[0].correlation_id if state.messages else self.correlation_id
            )
        self.messages = list(state.messages)
        node_timeout = timeout_seconds or task.budget.max_wall_seconds

        try:
            if not state.route_target:
                if self._cancelled(cancel_event):
                    return self._cancel(state, checkpoint, "cancelled before routing")
                active_router = router or TaskRouter(
                    RoutingMode.FIXED, fixed_target=RouteTarget.STANDARD
                )
                decision = await active_router.route(task, provider=routing_provider)
                self._record_routing(state, decision)
                checkpoint.save(state)
            executor, planner_provider, reviewer_provider = self._select_route_binding(
                state,
                checkpoint,
                default=RouteBinding(executor, planner_provider, reviewer_provider),
                bindings=route_bindings,
            )
            if state.current_node == "planner":
                if self._cancelled(cancel_event):
                    return self._cancel(state, checkpoint, "cancelled before planning")
                state.plan = await self._read_only_role_call(
                    planner_provider,
                    system=(
                        "You are the planner node. Produce a concise evidence-driven plan. "
                        "You cannot call tools or modify files."
                    ),
                    user=task.problem_statement,
                    timeout_seconds=node_timeout,
                )
                self._record(
                    state,
                    checkpoint,
                    "planner",
                    "executor",
                    state.plan,
                    permission="read",
                )
                state.current_node = "executor"
                checkpoint.save(state)

            while state.cycle < max_cycles:
                if self._cancelled(cancel_event):
                    return self._cancel(state, checkpoint, "cancelled before executor")
                if state.current_node == "executor":
                    state.cycle += 1
                    state.executor_run_id = f"{actual_run_id}_executor_{state.cycle}"
                    checkpoint.save(state)
                    effective_task = task
                    if state.review_feedback:
                        effective_task = replace(
                            task,
                            problem_statement=(
                                f"{task.problem_statement}\n\n"
                                f"Read-only reviewer feedback:\n{state.review_feedback}"
                            ),
                        )
                    executor_state = await self._run_executor(
                        executor,
                        effective_task,
                        run_id=state.executor_run_id,
                        timeout_seconds=node_timeout,
                        cancel_event=cancel_event,
                    )
                    state.final_answer = executor_state.final_answer
                    self._record(
                        state,
                        checkpoint,
                        "executor",
                        "verifier",
                        f"status={executor_state.status.value}; "
                        f"reason={executor_state.failure_reason or 'none'}",
                        permission="write",
                    )
                    if executor_state.status is RunStatus.CANCELLED:
                        return self._cancel(state, checkpoint, "executor cancelled")
                    if executor_state.status is not RunStatus.COMPLETED:
                        state.status = RunStatus.FAILED
                        state.failure_reason = (
                            executor_state.failure_reason or "executor did not complete"
                        )
                        checkpoint.save(state)
                        return state
                    # AgentRuntime only reaches COMPLETED after its deterministic verifier passes.
                    self._record(
                        state,
                        checkpoint,
                        "verifier",
                        "reviewer",
                        "deterministic verification passed",
                    )
                    state.current_node = "reviewer"
                    checkpoint.save(state)

                if self._cancelled(cancel_event):
                    return self._cancel(state, checkpoint, "cancelled before review")
                current = capture_text_snapshot(task.workspace, include_paths=task.allowed_paths)
                diff = self._snapshot_diff(baseline, current)
                review = await self._read_only_role_call(
                    reviewer_provider,
                    system=(
                        "You are the read-only reviewer node. Return JSON only: "
                        '{"verdict":"pass|replan","feedback":"..."}. '
                        "Do not call tools. Pass only when the verified patch addresses the task."
                    ),
                    user=(
                        f"Task:\n{task.problem_statement}\n\n"
                        f"Executor answer:\n{state.final_answer}\n\n"
                        f"Verified diff:\n{diff[:100_000]}"
                    ),
                    timeout_seconds=node_timeout,
                )
                verdict, feedback = self._parse_review(review)
                state.review_feedback = feedback
                if verdict == "pass":
                    self._record(state, checkpoint, "reviewer", "finish", feedback)
                    state.current_node = "finish"
                    state.status = RunStatus.COMPLETED
                    checkpoint.save(state)
                    return state
                self._record(state, checkpoint, "reviewer", "executor", feedback)
                state.current_node = "executor"
                checkpoint.save(state)

            state.status = RunStatus.FAILED
            state.failure_reason = "multi-agent review cycle budget exceeded"
            checkpoint.save(state)
            return state
        except TimeoutError:
            state.status = RunStatus.FAILED
            state.failure_reason = f"node timeout after {node_timeout:.1f} seconds"
            checkpoint.save(state)
            return state
        except asyncio.CancelledError:
            # Preserve RUNNING/current_node so a process-level interruption can resume.
            checkpoint.save(state)
            raise

    @staticmethod
    def _record_routing(state: MultiAgentRunState, decision: RoutingDecision) -> None:
        state.routing_mode = decision.mode.value
        state.route_target = decision.target.value
        state.routing_reason = decision.reason
        state.routing_confidence = decision.confidence
        state.routing_cascade = list(decision.cascade)

    @staticmethod
    def _select_route_binding(
        state: MultiAgentRunState,
        checkpoint: MultiAgentCheckpointStore,
        *,
        default: RouteBinding,
        bindings: dict[RouteTarget, RouteBinding] | None,
    ) -> tuple[RuntimeExecutor, ModelProvider, ModelProvider]:
        if bindings is None:
            return default.executor, default.planner_provider, default.reviewer_provider
        target = RouteTarget(state.route_target)
        selected = bindings.get(target)
        if selected is None:
            state.routing_reason = (
                f"{state.routing_reason}; binding_missing:{target.value}; fallback to standard"
            )
            state.routing_cascade.extend(
                (f"binding_missing:{target.value}", "binding_fallback:standard")
            )
            state.route_target = RouteTarget.STANDARD.value
            selected = bindings.get(RouteTarget.STANDARD, default)
            checkpoint.save(state)
        return selected.executor, selected.planner_provider, selected.reviewer_provider

    def _record(
        self,
        state: MultiAgentRunState,
        checkpoint: MultiAgentCheckpointStore,
        sender: str,
        recipient: str,
        content: str,
        *,
        permission: str = "read",
    ) -> None:
        message = self.send(
            sender,
            recipient,
            state.task_id,
            content,
            permission=permission,
        )
        state.messages.append(message)
        checkpoint.save(state)

    @staticmethod
    async def _read_only_role_call(
        provider: ModelProvider,
        *,
        system: str,
        user: str,
        timeout_seconds: float,
    ) -> str:
        request = ModelRequest(
            messages=(Message("system", system), Message("user", user)),
            tools=(),
            max_output_tokens=1024,
        )
        response = await asyncio.wait_for(provider.complete(request), timeout=timeout_seconds)
        if response.tool_calls:
            raise ValueError("read-only orchestration node attempted a tool call")
        return response.content

    @staticmethod
    async def _run_executor(
        executor: RuntimeExecutor,
        task: PublicTaskSpec,
        *,
        run_id: str,
        timeout_seconds: float,
        cancel_event: asyncio.Event | None,
    ) -> AgentState:
        execution = asyncio.create_task(executor.run(task, run_id=run_id, resume=True))
        if cancel_event is None:
            return await asyncio.wait_for(execution, timeout=timeout_seconds)
        cancellation = asyncio.create_task(cancel_event.wait())
        done, _ = await asyncio.wait(
            {execution, cancellation},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if execution in done:
            cancellation.cancel()
            return execution.result()
        execution.cancel()
        cancellation.cancel()
        with suppress(asyncio.CancelledError):
            await execution
        if cancel_event.is_set():
            return AgentState(run_id, task.task_id, status=RunStatus.CANCELLED)
        raise TimeoutError

    @staticmethod
    def _parse_review(content: str) -> tuple[str, str]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("reviewer response must be JSON") from error
        if not isinstance(parsed, dict):
            raise ValueError("reviewer response must be a JSON object")
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in {"pass", "replan"}:
            raise ValueError("reviewer verdict must be pass or replan")
        return verdict, str(parsed.get("feedback", "")).strip()

    @staticmethod
    def _snapshot_diff(before: dict[str, str], after: dict[str, str]) -> str:
        chunks: list[str] = []
        for path in sorted(set(before) | set(after)):
            if before.get(path) == after.get(path):
                continue
            chunks.extend(
                difflib.unified_diff(
                    before.get(path, "").splitlines(keepends=True),
                    after.get(path, "").splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
        return "".join(chunks)

    @staticmethod
    def _save_baseline(run_dir: Path, baseline: dict[str, str]) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "baseline.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _load_baseline(run_dir: Path) -> dict[str, str]:
        path = run_dir / "baseline.json"
        if not path.exists():
            raise ValueError("multi-agent baseline is missing")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
        ):
            raise ValueError("multi-agent baseline is invalid")
        return cast(dict[str, str], raw)

    @staticmethod
    def _cancelled(cancel_event: asyncio.Event | None) -> bool:
        return cancel_event is not None and cancel_event.is_set()

    @staticmethod
    def _cancel(
        state: MultiAgentRunState,
        checkpoint: MultiAgentCheckpointStore,
        reason: str,
    ) -> MultiAgentRunState:
        state.status = RunStatus.CANCELLED
        state.failure_reason = reason
        checkpoint.save(state)
        return state
