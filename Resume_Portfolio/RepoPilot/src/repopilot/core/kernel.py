"""Event-driven model and tool execution shared by v2 runtimes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from repopilot.core.contracts import (
    AgentState,
    ErrorType,
    ModelRequest,
    ModelResponse,
    PolicyOutcome,
    RunStatus,
    ToolCall,
    ToolResult,
)
from repopilot.core.events import RuntimeEvent, RuntimeEventKind
from repopilot.orchestration.parallel import execute_parallel_read_only
from repopilot.providers.base import (
    ModelProvider,
    ModelProviderError,
    StreamingModelProvider,
    empty_response_recovery_message,
)
from repopilot.runtime.cancellation import CancellationToken
from repopilot.runtime.checkpoint import ExecutionJournal
from repopilot.runtime.policy import ApprovalHandler, PolicyEngine, StaticApprovalHandler
from repopilot.tools.base import Tool, ToolContext, ToolRegistry
from repopilot.tools.contracts import StageScopedToolContracts
from repopilot.workspace.contracts import WorkspaceTask

EventSink = Callable[[RuntimeEvent], Awaitable[None] | None]
PersistState = Callable[[], None]


@dataclass(frozen=True, slots=True)
class AgentRuntimeConfig:
    """Stable execution knobs shared by task and session runtimes."""

    model_retries: int = 2
    tool_retries: int = 1
    parallel_read_tools: bool = True
    max_parallel_read_tools: int = 4
    stream_model_output: bool = False
    retry_base_seconds: float = 0.25
    retry_max_seconds: float = 2.0
    auto_finalize_after_successful_test: bool = False

    def __post_init__(self) -> None:
        if self.model_retries < 0 or self.tool_retries < 0:
            raise ValueError("retry counts must be non-negative")
        if self.retry_base_seconds < 0 or self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("retry delays must be non-negative and ordered")
        if not 1 <= self.max_parallel_read_tools <= 16:
            raise ValueError("max_parallel_read_tools must be between 1 and 16")


class AgentKernel:
    """Run model and tool transitions without owning a task or session lifecycle."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        registry: ToolRegistry,
        policy: PolicyEngine | None = None,
        approval: ApprovalHandler | None = None,
        config: AgentRuntimeConfig | None = None,
        tool_contracts: StageScopedToolContracts | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.policy = policy or PolicyEngine()
        self.approval = approval or StaticApprovalHandler(False)
        self.config = config or AgentRuntimeConfig()
        self.tool_contracts = tool_contracts

    async def request_model(
        self,
        request: ModelRequest,
        *,
        iteration: int,
        emit: EventSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ModelResponse:
        """Request the next action and emit lifecycle events for one model call."""

        if cancellation is not None:
            cancellation.raise_if_cancelled()
        await self._emit(
            emit,
            RuntimeEvent(
                RuntimeEventKind.TURN_STARTED,
                {"iteration": iteration, "message_count": len(request.messages)},
            ),
        )
        await self._emit(
            emit,
            RuntimeEvent(
                RuntimeEventKind.MODEL_CALL_STARTED,
                {"iteration": iteration, "message_count": len(request.messages)},
            ),
        )
        model_started = time.perf_counter()
        try:
            response = await self._call_model(request, cancellation=cancellation, emit=emit)
            if recovery_message := empty_response_recovery_message(response):
                raise ModelProviderError(recovery_message, recoverable=False)
        except ModelProviderError as error:
            await self._emit(
                emit,
                RuntimeEvent(RuntimeEventKind.TURN_FAILED, {"reason": str(error)}),
            )
            raise
        await self._emit(
            emit,
            RuntimeEvent(
                RuntimeEventKind.MODEL_CALL_COMPLETED,
                {
                    "model": response.model,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "tool_calls": len(response.tool_calls),
                    "duration_ms": round((time.perf_counter() - model_started) * 1000),
                },
            ),
        )
        for call in response.tool_calls:
            await self._emit(
                emit,
                RuntimeEvent(
                    RuntimeEventKind.TOOL_CALL_PROPOSED,
                    {"call_id": call.call_id, "tool": call.name},
                ),
            )
        if not response.tool_calls:
            await self._emit(
                emit,
                RuntimeEvent(RuntimeEventKind.TURN_COMPLETED, {"tool_calls": 0}),
            )
        return response

    async def execute_calls(
        self,
        calls: tuple[ToolCall, ...],
        *,
        task: WorkspaceTask,
        context: ToolContext,
        state: AgentState,
        journal: ExecutionJournal,
        persist: PersistState | None = None,
        emit: EventSink | None = None,
    ) -> tuple[ToolResult, ...]:
        """Execute a checkpointed batch of model-proposed calls in stable order."""

        results: list[ToolResult | None] = [None] * len(calls)
        if context.cancellation is not None:
            context.cancellation.raise_if_cancelled()
        executable: list[tuple[int, ToolCall, Tool]] = []
        for index, call in enumerate(calls):
            cached = journal.get(call.call_id)
            if cached is not None:
                results[index] = cached
                await self._emit_result(emit, cached)
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
            if self.tool_contracts is not None:
                contract = self.tool_contracts.validate(call, stage=state.workflow_stage)
                if not contract.allowed:
                    if contract.code is None:
                        raise RuntimeError("denied tool contract is missing a recovery code")
                    results[index] = ToolResult(
                        call.call_id,
                        call.name,
                        False,
                        error=f"{contract.code.value}: {contract.detail}",
                        error_type=ErrorType.VALIDATION,
                        recoverable=contract.recoverable,
                    )
                    continue
            decision = self.policy.decide(call, tool.spec, task)
            await self._emit(
                emit,
                RuntimeEvent(
                    RuntimeEventKind.POLICY_DECISION,
                    {
                        "call_id": call.call_id,
                        "tool": call.name,
                        "outcome": decision.outcome.value,
                        "reason": decision.reason,
                    },
                ),
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
                if persist is not None:
                    persist()
                await self._emit(
                    emit,
                    RuntimeEvent(
                        RuntimeEventKind.PERMISSION_REQUESTED,
                        {
                            "call_id": call.call_id,
                            "tool": call.name,
                            "reason": decision.reason,
                        },
                    ),
                )
                approved = await self.approval.approve(call, decision.reason)
                state.status = RunStatus.RUNNING
                if persist is not None:
                    persist()
                if not approved:
                    results[index] = ToolResult(
                        call.call_id,
                        call.name,
                        False,
                        error="human approval denied",
                        error_type=ErrorType.PERMISSION,
                    )
                    continue
                context.approved_call_ids.add(call.call_id)
            executable.append((index, call, tool))

        if (
            self.config.parallel_read_tools
            and len(executable) > 1
            and all(tool.spec.read_only for _, _, tool in executable)
        ):
            for _, call, _ in executable:
                await self._emit_started(emit, call)
            parallel_calls = tuple(call for _, call, _ in executable)
            parallel_results = await execute_parallel_read_only(
                parallel_calls,
                self.registry,
                context,
                max_concurrency=self.config.max_parallel_read_tools,
            )
            if context.cancellation is not None:
                context.cancellation.raise_if_cancelled()
            for (index, _, _), result in zip(executable, parallel_results, strict=True):
                state.tool_calls += 1
                results[index] = result
        else:
            for index, call, tool in executable:
                await self._emit_started(emit, call)
                result = await self._run_tool_with_retry(call, tool, context)
                if context.cancellation is not None:
                    context.cancellation.raise_if_cancelled()
                state.tool_calls += 1
                results[index] = result

        final: list[ToolResult] = []
        for index, final_result in enumerate(results):
            if final_result is None:
                raise AssertionError(f"tool call at index {index} produced no result")
            journal.put(final_result)
            await self._emit_result(emit, final_result)
            final.append(final_result)
        if self.tool_contracts is not None:
            for call, result in zip(calls, final, strict=True):
                state.workflow_stage = self.tool_contracts.next_stage(
                    state.workflow_stage, call, result
                )
        if persist is not None:
            persist()
        await self._emit(
            emit,
            RuntimeEvent(RuntimeEventKind.TURN_COMPLETED, {"tool_calls": len(final)}),
        )
        return tuple(final)

    async def _call_model(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken | None = None,
        emit: EventSink | None = None,
    ) -> ModelResponse:
        for attempt in range(self.config.model_retries + 1):
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            try:
                if self.config.stream_model_output and isinstance(
                    self.provider, StreamingModelProvider
                ):

                    async def on_text_delta(text: str) -> None:
                        if cancellation is not None:
                            cancellation.raise_if_cancelled()
                        await self._emit(
                            emit,
                            RuntimeEvent(RuntimeEventKind.MODEL_OUTPUT_DELTA, {"text": text}),
                        )

                    return await self.provider.complete_stream(request, on_text_delta=on_text_delta)
                return await self.provider.complete(request)
            except ModelProviderError as error:
                if not error.recoverable or attempt >= self.config.model_retries:
                    raise
                delay = error.retry_after_seconds
                if delay is None:
                    delay = min(
                        self.config.retry_base_seconds * (2**attempt),
                        self.config.retry_max_seconds,
                    )
                await self._emit(
                    emit,
                    RuntimeEvent(
                        RuntimeEventKind.MODEL_RETRYING,
                        {
                            "attempt": attempt + 1,
                            "delay_seconds": delay,
                            "status_code": error.status_code,
                        },
                    ),
                )
                await self._sleep_with_cancellation(delay, cancellation)
        raise AssertionError("unreachable")

    @staticmethod
    async def _sleep_with_cancellation(
        delay_seconds: float, cancellation: CancellationToken | None
    ) -> None:
        """Back off without making Ctrl+C wait for a provider retry delay."""
        if delay_seconds <= 0:
            return
        deadline = asyncio.get_running_loop().time() + delay_seconds
        while True:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 0.1))

    async def _run_tool_with_retry(
        self, call: ToolCall, tool: Tool, context: ToolContext
    ) -> ToolResult:
        for attempt in range(self.config.tool_retries + 1):
            if context.cancellation is not None:
                context.cancellation.raise_if_cancelled()
            result = await tool.run(call, context)
            if context.cancellation is not None:
                context.cancellation.raise_if_cancelled()
            if result.ok or not result.recoverable or not tool.spec.idempotent:
                return result
            if attempt >= self.config.tool_retries:
                return result
            await asyncio.sleep(0)
        raise AssertionError("unreachable")

    @staticmethod
    async def _emit(emit: EventSink | None, event: RuntimeEvent) -> None:
        if emit is None:
            return
        result = emit(event)
        if isinstance(result, Awaitable):
            await result

    async def _emit_started(self, emit: EventSink | None, call: ToolCall) -> None:
        await self._emit(
            emit,
            RuntimeEvent(
                RuntimeEventKind.TOOL_CALL_STARTED,
                {"call_id": call.call_id, "tool": call.name},
            ),
        )

    async def _emit_result(self, emit: EventSink | None, result: ToolResult) -> None:
        await self._emit(
            emit,
            RuntimeEvent(
                RuntimeEventKind.TOOL_CALL_COMPLETED,
                {
                    "call_id": result.call_id,
                    "tool": result.tool_name,
                    "ok": result.ok,
                    "error_type": result.error_type.value if result.error_type else None,
                    "side_effect": result.side_effect,
                    "cached": result.cached,
                },
            ),
        )
