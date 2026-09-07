from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from repopilot.core.contracts import ModelResponse
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.policy import PermissionMode, StaticApprovalHandler
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.coding import default_coding_tools
from repopilot.verification.engine import (
    VerificationCommand,
    VerificationEngine,
    VerificationKind,
    default_verification_commands,
    default_verification_plan,
)
from repopilot.workspace.project import ProjectWorkspace


def test_verification_engine_returns_structured_success_and_failure(tmp_path: Path) -> None:
    async def exercise() -> None:
        engine = VerificationEngine(LocalTrustedRunner(trusted=True))
        report = await engine.run(
            tmp_path,
            (
                VerificationCommand("success", (sys.executable, "-c", "print('verified')")),
                VerificationCommand("failure", (sys.executable, "-c", "raise SystemExit(3)")),
            ),
        )
        assert not report.ok
        assert report.results[0].ok
        assert not report.results[1].ok
        assert report.results[0].execution is not None
        assert report.results[0].execution.stdout == "verified\n"

    asyncio.run(exercise())


def test_session_verification_requires_approval_and_respects_plan_mode(tmp_path: Path) -> None:
    async def exercise() -> None:
        project = tmp_path / "project"
        project.mkdir()
        command = VerificationCommand("python", (sys.executable, "-c", "print('ok')"))
        approved_runtime = SessionRuntime(
            provider=ScriptedProvider([ModelResponse(content="unused")]),
            tools=default_coding_tools(),
            runner=LocalTrustedRunner(trusted=True),
            store=SessionStore(tmp_path / "sessions"),
            model="scripted",
            approval=StaticApprovalHandler(True),
        )
        metadata = approved_runtime.start(project)
        verified = await approved_runtime.verify(metadata, (command,))
        assert verified.report.ok
        assert verified.metadata.updated_at >= metadata.updated_at
        history = approved_runtime.verification_history(verified.metadata)
        assert len(history) == 1
        assert history[0].ok
        assert history[0].results[0].verification.label == "python"

        plan_runtime = SessionRuntime(
            provider=ScriptedProvider([ModelResponse(content="unused")]),
            tools=default_coding_tools(),
            runner=LocalTrustedRunner(trusted=True),
            store=SessionStore(tmp_path / "plan-sessions"),
            model="scripted",
            permission_mode=PermissionMode.PLAN,
            approval=StaticApprovalHandler(True),
        )
        plan_metadata = plan_runtime.start(project)
        blocked = await plan_runtime.verify(plan_metadata, (command,))
        assert not blocked.report.ok
        assert blocked.report.results[0].error == "plan mode permits read-only tools only"

    asyncio.run(exercise())


def test_default_verification_commands_cover_marker_based_languages(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        '{"scripts":{"test":"vitest","lint":"eslint .","build":"vite build"}}',
        encoding="utf-8",
    )
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    commands = default_verification_commands(ProjectWorkspace.discover(tmp_path))
    assert [command.command for command in commands] == [
        ("npm", "run", "test"),
        ("npm", "run", "lint"),
        ("npm", "run", "build"),
        ("go", "test", "./..."),
    ]
    assert [command.kind for command in commands] == [
        VerificationKind.TEST,
        VerificationKind.LINT,
        VerificationKind.BUILD,
        VerificationKind.TEST,
    ]


def test_verification_plan_selects_discovered_python_quality_checks(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'example'\nversion = '0.1.0'\n"
        "[tool.ruff]\nline-length = 100\n[tool.mypy]\nstrict = true\n",
        encoding="utf-8",
    )
    plan = default_verification_plan(ProjectWorkspace.discover(tmp_path))
    assert [command.label for command in plan.commands] == ["pytest", "ruff", "mypy"]
    assert [command.label for command in plan.select("test")] == ["pytest"]
    assert [command.label for command in plan.select("lint")] == ["ruff"]
    assert [command.label for command in plan.select("MYPY")] == ["mypy"]


def test_failed_verification_becomes_repair_evidence(tmp_path: Path) -> None:
    async def exercise() -> None:
        project = tmp_path / "project"
        project.mkdir()
        provider = ScriptedProvider([ModelResponse(content="I will repair the failed check.")])
        runtime = SessionRuntime(
            provider=provider,
            tools=default_coding_tools(),
            runner=LocalTrustedRunner(trusted=True),
            store=SessionStore(tmp_path / "sessions"),
            model="scripted",
            approval=StaticApprovalHandler(True),
        )
        metadata = runtime.start(project)
        failed = await runtime.verify(
            metadata,
            (VerificationCommand("fails", (sys.executable, "-c", "raise SystemExit(2)")),),
        )
        assert not failed.report.ok
        state = runtime.store.checkpoint(failed.metadata).load()
        assert state is not None
        assert state.messages[-1].name == "verification"
        assert "raise SystemExit" not in state.messages[-1].content
        evidence = runtime.evidence(failed.metadata)
        assert evidence.repair_ready
        assert evidence.latest_verification is not None
        assert evidence.latest_verification.results[0].execution is not None
        assert evidence.latest_verification.results[0].execution.exit_code == 2
        repaired = await runtime.repair(failed.metadata)
        assert repaired.answer == "I will repair the failed check."
        sent_context = "\n".join(
            message.content for request in provider.requests for message in request.messages
        )
        assert "Repair the latest persisted failed verification" in sent_context
        assert "raise SystemExit" not in sent_context

    asyncio.run(exercise())
