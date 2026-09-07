"""Bounded ReAct-style runtime with policy, checkpoint, retry, and verification."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast
from uuid import uuid4

from repopilot.context.builder import ContextBuilder
from repopilot.core.budgets import BudgetGuard
from repopilot.core.contracts import (
    AgentState,
    ErrorType,
    Message,
    ModelResponse,
    PolicyOutcome,
    RunStatus,
    ToolCall,
    ToolResult,
)
from repopilot.core.events import RuntimeEvent, RuntimeEventKind
from repopilot.core.kernel import AgentKernel, AgentRuntimeConfig
from repopilot.memory.store import Episode, EpisodicMemoryStore, SessionMemory
from repopilot.observability.trace import TraceRecorder
from repopilot.orchestration.parallel import execute_parallel_read_only
from repopilot.orchestration.planner import Plan, SimplePlanner
from repopilot.providers.base import ModelProvider, ModelProviderError
from repopilot.runtime.checkpoint import CheckpointStore, ExecutionJournal
from repopilot.runtime.policy import ApprovalHandler, PolicyEngine, StaticApprovalHandler
from repopilot.runtime.runner import CommandRunner
from repopilot.skills.registry import SkillRegistry
from repopilot.task import PublicTaskSpec
from repopilot.tools.base import Tool, ToolContext, ToolRegistry
from repopilot.tools.coding import capture_text_snapshot
from repopilot.tools.contracts import StageScopedToolContracts
from repopilot.verification.verifier import DeterministicVerifier


class AgentRuntime:
    """A single controlling agent with deterministic environment boundaries."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        tools: list[Tool],
        runner: CommandRunner,
        artifacts_root: Path,
        context_builder: ContextBuilder | None = None,
        policy: PolicyEngine | None = None,
        approval: ApprovalHandler | None = None,
        skill_registry: SkillRegistry | None = None,
        episodic_memory: EpisodicMemoryStore | None = None,
        planner: SimplePlanner | None = None,
        config: AgentRuntimeConfig | None = None,
        tool_contracts: StageScopedToolContracts | None = None,
    ) -> None:
        self.provider = provider
        self.registry = ToolRegistry(tools)
        self.runner = runner
        self.artifacts_root = artifacts_root
        self.context_builder = context_builder or ContextBuilder()
        self.policy = policy or PolicyEngine()
        self.approval = approval or StaticApprovalHandler(False)
        self.skill_registry = skill_registry
        self.episodic_memory = episodic_memory
        self.planner = planner or SimplePlanner()
        self.config = config or AgentRuntimeConfig()
        self.tool_contracts = tool_contracts
        self.kernel = AgentKernel(
            provider=self.provider,
            registry=self.registry,
            policy=self.policy,
            approval=self.approval,
            config=self.config,
            tool_contracts=self.tool_contracts,
        )

    async def run(
        self, task: PublicTaskSpec, *, run_id: str | None = None, resume: bool = True
    ) -> AgentState:
        if not task.workspace.is_dir():
            raise ValueError(f"task workspace does not exist: {task.workspace}")
        actual_run_id = run_id or f"run_{uuid4().hex[:12]}"
        run_dir = self.artifacts_root / actual_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = CheckpointStore(run_dir / "checkpoint.json")
        journal = ExecutionJournal(run_dir / "tool_journal.json")
        trace = TraceRecorder(run_dir / "events.jsonl")
        state = checkpoint.load() if resume else None
        was_resumed = state is not None
        if state is not None and (state.task_id != task.task_id or state.run_id != actual_run_id):
            raise ValueError("checkpoint identity does not match the requested run")
        if state is None:
            state = AgentState(actual_run_id, task.task_id, status=RunStatus.PREPARING)
            plan = self.planner.create(task)
            state.plan = plan.objectives()
            checkpoint.save(state)
        else:
            plan = Plan(tuple(), revision=1)
        baseline = self._baseline(run_dir, task)
        context = ToolContext(task, self.runner, baseline)
        verifier = DeterministicVerifier(self.runner, baseline)
        session_memory = SessionMemory()
        guard = BudgetGuard(task.budget)
        selected_skills = (
            self.skill_registry.select(task.problem_statement) if self.skill_registry else ()
        )
        trace.record(
            "run_started",
            {
                "run_id": state.run_id,
                "task_id": task.task_id,
                "resumed": was_resumed,
            },
        )
        if state.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return state
        state.status = RunStatus.RUNNING
        checkpoint.save(state)

        last_changed: list[str] | None = None
        consecutive_no_change = 0

        def emit(event: RuntimeEvent) -> None:
            self._record_kernel_event(trace, event)

        while state.status is RunStatus.RUNNING:
            reason = guard.exceeded_reason(state)
            if reason is not None:
                self._fail(state, reason, trace, checkpoint)
                break
            if state.pending_calls:
                results = await self.kernel.execute_calls(
                    tuple(state.pending_calls),
                    task=task,
                    context=context,
                    state=state,
                    journal=journal,
                    persist=lambda: checkpoint.save(state),
                    emit=emit,
                )
                state.pending_calls.clear()
                for result in results:
                    context.recent_results.append(result)
                    state.messages.append(
                        Message(
                            "tool",
                            json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True),
                            name=result.tool_name,
                            tool_call_id=result.call_id,
                        )
                    )
                    if not result.ok:
                        session_memory.remember(
                            f"{result.tool_name} failed with {result.error_type}: {result.error}"
                        )
                checkpoint.save(state)
                if self._can_finalize_after_successful_test(task, context, results):
                    verification = await verifier.verify(task)
                    trace.record(
                        "verification_finished",
                        {
                            "passed": verification.passed,
                            "summary": verification.summary,
                            "details": verification.details,
                        },
                    )
                    if verification.passed:
                        state.final_answer = (
                            "Applied the requested patch and verified the immutable visible tests."
                        )
                        state.messages.append(Message("assistant", state.final_answer))
                        state.status = RunStatus.COMPLETED
                        checkpoint.save(state)
                        if self.episodic_memory is not None:
                            self.episodic_memory.append(
                                Episode(
                                    task.task_id,
                                    task.problem_statement,
                                    "; ".join(state.plan),
                                    "completed",
                                )
                            )
                        break
                continue

            state.iteration += 1
            episodes = (
                tuple(self.episodic_memory.search(task.problem_statement))
                if self.episodic_memory is not None
                else ()
            )
            request = self.context_builder.build(
                task=task,
                state=state,
                tools=(
                    self.tool_contracts.visible_specs(state.workflow_stage)
                    if self.tool_contracts is not None
                    else self.registry.specs()
                ),
                session_memory=session_memory,
                episodes=episodes,
                skills=selected_skills,
            )
            try:
                response = await self.kernel.request_model(
                    request, iteration=state.iteration, emit=emit
                )
            except ModelProviderError as error:
                self._fail(state, f"model provider failed: {error}", trace, checkpoint)
                break
            state.input_tokens += response.usage.input_tokens
            state.output_tokens += response.usage.output_tokens
            if response.tool_calls:
                state.messages.append(
                    Message(
                        "assistant",
                        "Requested tools: " + ", ".join(call.name for call in response.tool_calls),
                        tool_calls=response.tool_calls,
                    )
                )
                state.pending_calls = list(response.tool_calls)
                checkpoint.save(state)
                continue

            state.final_answer = self._final_content(response)
            state.messages.append(Message("assistant", state.final_answer))
            state.status = RunStatus.VERIFYING
            checkpoint.save(state)
            verification = await verifier.verify(task)
            trace.record(
                "verification_finished",
                {
                    "passed": verification.passed,
                    "summary": verification.summary,
                    "details": verification.details,
                },
            )
            if verification.passed:
                state.status = RunStatus.COMPLETED
                checkpoint.save(state)
                if self.episodic_memory is not None:
                    self.episodic_memory.append(
                        Episode(
                            task.task_id, task.problem_statement, "; ".join(state.plan), "completed"
                        )
                    )
                break
            if verification.recoverable and guard.exceeded_reason(state) is None:
                raw_changed = verification.details.get("changed_files", [])
                current_changed = list(raw_changed) if isinstance(raw_changed, list) else []
                if current_changed == last_changed:
                    consecutive_no_change += 1
                else:
                    consecutive_no_change = 0
                    last_changed = current_changed
                if consecutive_no_change >= 2:
                    self._fail(
                        state,
                        "repeated verification failure with no new file changes",
                        trace,
                        checkpoint,
                    )
                    break
                session_memory.remember(verification.summary)
                state.messages.append(
                    Message(
                        "tool",
                        "[DETERMINISTIC VERIFIER]\n"
                        + json.dumps(verification.details, ensure_ascii=False, sort_keys=True),
                        name="verifier",
                    )
                )
                plan = self.planner.replan(plan, verification.summary)
                state.plan = plan.objectives()
                state.status = RunStatus.RUNNING
                checkpoint.save(state)
                continue
            self._fail(state, verification.summary, trace, checkpoint)

        trace.record(
            "run_finished",
            {
                "status": state.status.value,
                "iterations": state.iteration,
                "tool_calls": state.tool_calls,
                "failure_reason": state.failure_reason,
            },
        )
        return state

    def _can_finalize_after_successful_test(
        self,
        task: PublicTaskSpec,
        context: ToolContext,
        results: tuple[ToolResult, ...],
    ) -> bool:
        """Allow the synthetic diagnostic runner to avoid an unneeded model turn.

        This is intentionally disabled by default for normal coding sessions.
        It applies only to trusted public fixtures after one successful exact
        patch and the fixture's immutable test command have both succeeded.
        """

        if not self.config.auto_finalize_after_successful_test or not task.trusted_fixture:
            return False
        tests_passed = any(result.tool_name == "run_tests" and result.ok for result in results)
        patch_applied = any(
            result.tool_name == "apply_patch" and result.ok for result in context.recent_results
        )
        return tests_passed and patch_applied

    @staticmethod
    def _record_kernel_event(trace: TraceRecorder, event: RuntimeEvent) -> None:
        """Preserve the v1 trace names while recording the richer kernel stream."""

        if event.kind is RuntimeEventKind.MODEL_CALL_COMPLETED:
            trace.record("model_call_finished", event.data)
        elif event.kind is RuntimeEventKind.TOOL_CALL_COMPLETED and bool(event.data.get("cached")):
            trace.record(
                "tool_result_replayed",
                {"call_id": event.data["call_id"], "tool": event.data["tool"]},
            )
        elif event.kind is RuntimeEventKind.TOOL_CALL_COMPLETED:
            trace.record("tool_call_finished", event.data)
        else:
            trace.record(event.kind.value, event.data)

    async def _call_model(self, request: object) -> ModelResponse:
        from repopilot.core.contracts import ModelRequest

        typed_request = cast(ModelRequest, request)
        for attempt in range(self.config.model_retries + 1):
            try:
                return await self.provider.complete(typed_request)
            except ModelProviderError as error:
                if not error.recoverable or attempt >= self.config.model_retries:
                    raise
                await asyncio.sleep(0)
        raise AssertionError("unreachable")

    async def _execute_calls(
        self,
        calls: tuple[ToolCall, ...],
        task: PublicTaskSpec,
        context: ToolContext,
        state: AgentState,
        journal: ExecutionJournal,
        trace: TraceRecorder,
        checkpoint: CheckpointStore,
    ) -> tuple[ToolResult, ...]:
        results: list[ToolResult | None] = [None] * len(calls)
        executable: list[tuple[int, ToolCall, Tool]] = []
        for index, call in enumerate(calls):
            cached = journal.get(call.call_id)
            if cached is not None:
                results[index] = cached
                trace.record("tool_result_replayed", {"call_id": call.call_id, "tool": call.name})
                continue
            tool = self.registry.get(call.name)
            if tool is None:
                results[index] = ToolResult(
                    call.call_id,
                    call.name,
                    False,
                    error="unknown tool",
                    error_type=ErrorType.VALIDATION,
                )
                continue
            decision = self.policy.decide(call, tool.spec, task)
            trace.record(
                "policy_decision",
                {
                    "call_id": call.call_id,
                    "tool": call.name,
                    "outcome": decision.outcome.value,
                    "reason": decision.reason,
                },
            )
            if decision.outcome is PolicyOutcome.DENY:
                results[index] = ToolResult(
                    call.call_id,
                    call.name,
                    False,
                    error=decision.reason,
                    error_type=ErrorType.PERMISSION,
                )
                continue
            if decision.outcome is PolicyOutcome.REQUIRE_APPROVAL:
                state.status = RunStatus.APPROVAL_REQUIRED
                checkpoint.save(state)
                approved = await self.approval.approve(call, decision.reason)
                state.status = RunStatus.RUNNING
                if not approved:
                    results[index] = ToolResult(
                        call.call_id,
                        call.name,
                        False,
                        error="human approval denied",
                        error_type=ErrorType.PERMISSION,
                    )
                    continue
            executable.append((index, call, tool))

        if (
            self.config.parallel_read_tools
            and len(executable) > 1
            and all(tool.spec.read_only for _, _, tool in executable)
        ):
            parallel_calls = tuple(call for _, call, _ in executable)
            parallel_results = await execute_parallel_read_only(
                parallel_calls, self.registry, context
            )
            for (index, _, _), result in zip(executable, parallel_results, strict=True):
                state.tool_calls += 1
                results[index] = result
        else:
            for index, call, tool in executable:
                result = await self._run_tool_with_retry(call, tool, context)
                state.tool_calls += 1
                results[index] = result

        final: list[ToolResult] = []
        for index, final_result in enumerate(results):
            if final_result is None:
                raise AssertionError(f"tool call at index {index} produced no result")
            journal.put(final_result)
            trace.record(
                "tool_call_finished",
                {
                    "call_id": final_result.call_id,
                    "tool": final_result.tool_name,
                    "ok": final_result.ok,
                    "error_type": (
                        final_result.error_type.value if final_result.error_type else None
                    ),
                    "side_effect": final_result.side_effect,
                    "cached": final_result.cached,
                },
            )
            final.append(final_result)
        checkpoint.save(state)
        return tuple(final)

    async def _run_tool_with_retry(
        self, call: ToolCall, tool: Tool, context: ToolContext
    ) -> ToolResult:
        for attempt in range(self.config.tool_retries + 1):
            result = await tool.run(call, context)
            if result.ok or not result.recoverable or not tool.spec.idempotent:
                return result
            if attempt >= self.config.tool_retries:
                return result
            await asyncio.sleep(0)
        raise AssertionError("unreachable")

    @staticmethod
    def _final_content(response: ModelResponse) -> str:
        try:
            value = json.loads(response.content)
        except json.JSONDecodeError:
            return response.content
        if isinstance(value, dict) and value.get("type") == "finish":
            return str(value.get("answer", ""))
        return response.content

    @staticmethod
    def _fail(
        state: AgentState,
        reason: str,
        trace: TraceRecorder,
        checkpoint: CheckpointStore,
    ) -> None:
        state.status = RunStatus.FAILED
        state.failure_reason = reason
        checkpoint.save(state)
        trace.record("run_failed", {"reason": reason})

    @staticmethod
    def _baseline(run_dir: Path, task: PublicTaskSpec) -> dict[str, str]:
        path = run_dir / "baseline.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("baseline artifact must be an object")
            return {str(key): str(value) for key, value in raw.items()}
        baseline = capture_text_snapshot(task.workspace, include_paths=task.allowed_paths)
        path.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return baseline
