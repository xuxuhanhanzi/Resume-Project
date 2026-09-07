"""Bounded read-only background analysts for an interactive RepoPilot session."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

from repopilot.core.contracts import Message, ModelRequest
from repopilot.providers.base import ModelProvider, ModelProviderError


@dataclass(frozen=True, slots=True)
class BackgroundAgentSnapshot:
    task_id: str
    description: str
    running: bool
    result: str | None
    error: str | None


class BackgroundAgentManager:
    """Run at most two analysis-only model calls without workspace or tool authority."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider
        self._tasks: dict[str, tuple[str, asyncio.Task[str]]] = {}

    def start(self, *, description: str, question: str, evidence: str) -> str:
        if len([task for _description, task in self._tasks.values() if not task.done()]) >= 2:
            raise RuntimeError("at most two background subagents may run in one session")
        if not 1 <= len(description.strip()) <= 120:
            raise ValueError("subagent description must contain 1-120 characters")
        if not 1 <= len(question.strip()) <= 5_000 or not 1 <= len(evidence.strip()) <= 60_000:
            raise ValueError("subagent question or evidence is outside the allowed size")
        task_id = f"agent_{uuid4().hex[:12]}"
        task = asyncio.create_task(self._run(question=question, evidence=evidence))
        self._tasks[task_id] = (description.strip(), task)
        return task_id

    @property
    def has_running(self) -> bool:
        return any(not task.done() for _description, task in self._tasks.values())

    async def snapshot(self, task_id: str) -> BackgroundAgentSnapshot:
        record = self._tasks.get(task_id)
        if record is None:
            raise ValueError("unknown background subagent")
        description, task = record
        if not task.done():
            return BackgroundAgentSnapshot(task_id, description, True, None, None)
        try:
            result = task.result()
        except asyncio.CancelledError:
            return BackgroundAgentSnapshot(task_id, description, False, None, "cancelled")
        except (ModelProviderError, RuntimeError, ValueError) as error:
            return BackgroundAgentSnapshot(task_id, description, False, None, str(error))
        return BackgroundAgentSnapshot(task_id, description, False, result, None)

    async def snapshots(self) -> tuple[BackgroundAgentSnapshot, ...]:
        return tuple(await asyncio.gather(*(self.snapshot(task_id) for task_id in self._tasks)))

    async def stop(self, task_id: str) -> BackgroundAgentSnapshot:
        record = self._tasks.get(task_id)
        if record is None:
            raise ValueError("unknown background subagent")
        _description, task = record
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return await self.snapshot(task_id)

    async def aclose(self) -> None:
        running = [task for _description, task in self._tasks.values() if not task.done()]
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    async def _run(self, *, question: str, evidence: str) -> str:
        response = await self.provider.complete(
            ModelRequest(
                messages=(
                    Message(
                        "system",
                        "You are RepoPilot's background analysis agent. Analyze only the "
                        "provided untrusted evidence. You have no tools, no workspace access, "
                        "and no authority to claim an action was performed. Return concise "
                        "findings, evidence, and open questions.",
                    ),
                    Message(
                        "user",
                        f"Question:\n{question}\n\n[UNTRUSTED EVIDENCE]\n{evidence}",
                    ),
                ),
                tools=(),
                max_output_tokens=768,
            )
        )
        if response.tool_calls:
            raise RuntimeError("background subagent attempted a forbidden tool call")
        return response.content.strip() or "Background subagent returned no analysis."
