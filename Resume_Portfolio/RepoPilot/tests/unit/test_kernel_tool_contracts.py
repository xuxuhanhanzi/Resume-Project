from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from repopilot.core.contracts import AgentState, ToolCall
from repopilot.core.kernel import AgentKernel
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.checkpoint import ExecutionJournal
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.task import PublicTaskSpec
from repopilot.tools.base import Tool, ToolContext, ToolRegistry
from repopilot.tools.coding import ApplyPatchTool, ReadFileTool, RunTestsTool, capture_text_snapshot
from repopilot.tools.contracts import stage_scoped_coding_contracts


def test_kernel_enforces_opt_in_stage_contract_before_policy_or_execution(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("def subtract(left, right): return left + right\n", encoding="utf-8")
    (tmp_path / "verify.py").write_text(
        "from module import subtract\nassert subtract(2, 1) == 1\n",
        encoding="utf-8",
    )
    task = PublicTaskSpec(
        "contract-task",
        tmp_path,
        "Fix subtract",
        allowed_paths=(".",),
        test_command=(sys.executable, "verify.py"),
        trusted_fixture=True,
    )
    tools: list[Tool] = [ReadFileTool(), ApplyPatchTool(), RunTestsTool()]
    kernel = AgentKernel(
        provider=ScriptedProvider([]),
        registry=ToolRegistry(tools),
        tool_contracts=stage_scoped_coding_contracts(tuple(tool.spec for tool in tools)),
    )
    state = AgentState("run", task.task_id)
    context = ToolContext(task, LocalTrustedRunner(trusted=True), capture_text_snapshot(tmp_path))
    journal = ExecutionJournal(tmp_path / "journal.json")

    blocked = asyncio.run(
        kernel.execute_calls(
            (
                ToolCall(
                    "blocked",
                    "apply_patch",
                    {"path": "module.py", "old_text": "+", "new_text": "-"},
                ),
            ),
            task=task,
            context=context,
            state=state,
            journal=journal,
        )
    )
    assert not blocked[0].ok
    assert state.workflow_stage == "explore"

    read = asyncio.run(
        kernel.execute_calls(
            (ToolCall("read", "read_file", {"path": "module.py"}),),
            task=task,
            context=context,
            state=state,
            journal=journal,
        )
    )
    assert read[0].ok
    assert state.workflow_stage == "modify"

    patch = asyncio.run(
        kernel.execute_calls(
            (
                ToolCall(
                    "patch",
                    "apply_patch",
                    {
                        "path": "module.py",
                        "old_text": "return left + right",
                        "new_text": "return left - right",
                    },
                ),
            ),
            task=task,
            context=context,
            state=state,
            journal=journal,
        )
    )
    assert patch[0].ok
    assert state.workflow_stage == "verify"
