from __future__ import annotations

import asyncio
from pathlib import Path

from repopilot.core.contracts import ModelResponse, ToolCall
from repopilot.mcp.protocol import InProcessMCPTransport, MCPClient, MCPRemoteTool, MCPServer
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
