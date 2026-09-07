from __future__ import annotations

import asyncio
from pathlib import Path

from repopilot.core.contracts import ModelResponse, Permission, ToolCall, ToolResult, ToolSpec
from repopilot.mcp.protocol import (
    InProcessMCPTransport,
    MCPClient,
    MCPPrompt,
    MCPRemoteTool,
    MCPResource,
    MCPServer,
)
from repopilot.orchestration.parallel import execute_parallel_read_only
from repopilot.orchestration.reviewer import ReviewerTool
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.task import PublicTaskSpec
from repopilot.tools.base import ToolContext, ToolRegistry
from repopilot.tools.coding import ApplyPatchTool, ListFilesTool, ReadFileTool


def _context(tmp_path: Path) -> ToolContext:
    (tmp_path / "a.py").write_text("value = 1\n", encoding="utf-8")
    task = PublicTaskSpec("t", tmp_path, "inspect", trusted_fixture=True)
    return ToolContext(task, LocalTrustedRunner(trusted=True))


def test_mcp_discovery_call_and_remote_adapter(tmp_path: Path) -> None:
    context = _context(tmp_path)
    server = MCPServer(ToolRegistry([ListFilesTool()]), context)
    client = MCPClient(InProcessMCPTransport(server))

    async def exercise() -> None:
        initialized = await client.initialize()
        specs = await client.list_tools()
        direct = await client.call_tool(ToolCall("c1", "list_files", {"path": "."}))
        remote = await MCPRemoteTool(client, specs[0]).run(
            ToolCall("c2", "list_files", {"path": "."}), context
        )
        assert initialized["capabilities"]
        assert specs[0].name == "list_files"
        assert direct.ok and remote.ok

    asyncio.run(exercise())


def test_mcp_discovers_bounded_resources_and_prompts(tmp_path: Path) -> None:
    context = _context(tmp_path)
    server = MCPServer(
        ToolRegistry([ListFilesTool()]),
        context,
        resources=(
            MCPResource(
                "repo://guide",
                "Repository guide",
                "Run focused tests before a final answer.",
                "A small project workflow note.",
            ),
        ),
        prompts=(
            MCPPrompt(
                "review-change",
                "Review {{path}} for regressions.",
                "Produce a focused code review prompt.",
                ("path",),
            ),
        ),
    )
    client = MCPClient(InProcessMCPTransport(server))

    async def exercise() -> None:
        await client.initialize()
        resources = await client.list_resources()
        contents = await client.read_resource("repo://guide")
        prompts = await client.list_prompts()
        messages = await client.get_prompt("review-change", arguments={"path": "src/app.py"})
        assert resources[0].name == "Repository guide"
        assert contents[0].text.startswith("Run focused tests")
        assert prompts[0].arguments == ("path",)
        assert messages[0].text == "Review src/app.py for regressions."

    asyncio.run(exercise())


def test_parallel_executor_rejects_side_effecting_tool(tmp_path: Path) -> None:
    context = _context(tmp_path)
    registry = ToolRegistry([ReadFileTool(), ApplyPatchTool()])
    results = asyncio.run(
        execute_parallel_read_only(
            (
                ToolCall("r", "read_file", {"path": "a.py"}),
                ToolCall(
                    "p",
                    "apply_patch",
                    {"path": "a.py", "old_text": "1", "new_text": "2"},
                ),
            ),
            registry,
            context,
        )
    )

    assert results[0].ok
    assert not results[1].ok
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "value = 1\n"


def test_parallel_executor_contains_one_read_failure_without_losing_siblings(
    tmp_path: Path,
) -> None:
    class FailingReadTool:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec("failing_read", "fails predictably", {}, permission=Permission.READ)

        async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
            del call, context
            raise RuntimeError("fixture failure")

    context = _context(tmp_path)
    registry = ToolRegistry([FailingReadTool(), ListFilesTool()])
    results = asyncio.run(
        execute_parallel_read_only(
            (
                ToolCall("bad", "failing_read", {}),
                ToolCall("good", "list_files", {"path": "."}),
            ),
            registry,
            context,
            max_concurrency=2,
        )
    )

    assert not results[0].ok
    assert "fixture failure" in (results[0].error or "")
    assert results[1].ok


def test_reviewer_is_read_only_agent_as_tool(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [ModelResponse(content='{"verdict":"pass","issues":[],"confidence":0.9}')]
    )
    result = asyncio.run(
        ReviewerTool(provider).run(
            ToolCall("review", "review_diff", {"diff": "+ return 1", "problem": "fix"}),
            _context(tmp_path),
        )
    )

    assert result.ok
    assert "pass" in str(result.data)
    assert provider.requests[0].tools == ()
