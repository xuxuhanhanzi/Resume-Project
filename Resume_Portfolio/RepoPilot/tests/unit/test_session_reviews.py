from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from repopilot.core.contracts import ModelResponse
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.runner import ExecutionResult
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.coding import default_coding_tools


class _DiffRunner:
    def __init__(self, diff: str) -> None:
        self.diff = diff
        self.commands: list[tuple[str, ...]] = []

    async def run(self, command: Sequence[str], **_: object) -> ExecutionResult:
        normalized = tuple(command)
        self.commands.append(normalized)
        return ExecutionResult(normalized, 0, self.diff, "")


def _runtime(tmp_path: Path, provider: ScriptedProvider, runner: _DiffRunner) -> SessionRuntime:
    return SessionRuntime(
        provider=provider,
        tools=default_coding_tools(),
        runner=runner,
        store=SessionStore(tmp_path / "sessions"),
        model="scripted",
    )


def test_quality_review_is_read_only_and_persists_a_structured_observation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    diff = "--- a/module.py\n+++ b/module.py\n@@ -1 +1 @@\n-old\n+new\n"
    provider = ScriptedProvider(
        [ModelResponse(content='{"verdict":"needs-attention","issues":["add a test"]}')]
    )
    runner = _DiffRunner(diff)
    runtime = _runtime(tmp_path, provider, runner)

    reviewed = asyncio.run(runtime.review_working_tree(runtime.start(project)))

    assert reviewed.review.ok
    assert reviewed.review.data == {"verdict": "needs-attention", "issues": ["add a test"]}
    assert runner.commands == [("git", "diff", "--no-ext-diff", "--unified=3", "--")]
    assert provider.requests[0].tools == ()
    assert "read-only code reviewer" in provider.requests[0].messages[0].content
    state = runtime.store.checkpoint(reviewed.metadata).load()
    assert state is not None
    assert state.messages[-1].name == "review"
    trace = runtime.trace(reviewed.metadata)
    assert [event["kind"] for event in trace[-2:]] == ["review_started", "review_completed"]


def test_security_review_uses_a_focused_prompt_and_empty_diff_stays_local(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    provider = ScriptedProvider([ModelResponse(content='{"verdict":"pass","findings":[]}')])
    runtime = _runtime(
        tmp_path,
        provider,
        _DiffRunner("--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-old\n+new\n"),
    )

    reviewed = asyncio.run(runtime.review_working_tree(runtime.start(project), mode="security"))

    assert reviewed.review.ok
    assert "application-security reviewer" in provider.requests[0].messages[0].content
    assert "confidence from 1 to 10" in provider.requests[0].messages[0].content

    empty_provider = ScriptedProvider([])
    empty_runtime = _runtime(tmp_path, empty_provider, _DiffRunner(""))
    empty = asyncio.run(empty_runtime.review_working_tree(empty_runtime.start(project)))

    assert empty.review.ok
    assert empty.review.data["verdict"] == "clean"
    assert empty_provider.requests == []
