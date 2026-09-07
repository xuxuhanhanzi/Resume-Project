from __future__ import annotations

import asyncio
from pathlib import Path

from repopilot.agents.tools import ExploreAgentTool, TestAgentTool
from repopilot.core.contracts import ModelResponse, ToolCall
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.tools.base import ToolContext
from repopilot.workspace.contracts import InteractiveTask


def test_subagents_receive_evidence_only_and_cannot_call_tools(tmp_path: Path) -> None:
    task = InteractiveTask("session", tmp_path, "Analyze selected evidence")
    context = ToolContext(task, LocalTrustedRunner(trusted=True))
    provider = ScriptedProvider([ModelResponse(content='{"findings":["parser is shared"]}')])

    async def exercise() -> None:
        result = await ExploreAgentTool(provider).run(
            ToolCall(
                "explore",
                "explore_code",
                {"question": "Where is the parser shared?", "evidence": "src/parser.py"},
            ),
            context,
        )
        assert result.ok
        assert result.data == {"findings": ["parser is shared"]}
        request = provider.requests[0]
        assert request.tools == ()
        assert "UNTRUSTED EVIDENCE" in request.messages[1].content

    asyncio.run(exercise())


def test_subagent_rejects_nested_tool_requests(tmp_path: Path) -> None:
    task = InteractiveTask("session", tmp_path, "Analyze test output")
    context = ToolContext(task, LocalTrustedRunner(trusted=True))
    provider = ScriptedProvider(
        [ModelResponse(tool_calls=(ToolCall("nested", "run_shell", {"command": ["echo"]}),))]
    )

    async def exercise() -> None:
        result = await TestAgentTool(provider).run(
            ToolCall(
                "tests",
                "analyze_tests",
                {"question": "Why did tests fail?", "evidence": "AssertionError at test_parser"},
            ),
            context,
        )
        assert not result.ok
        assert result.error == "subagent attempted a tool call"

    asyncio.run(exercise())
