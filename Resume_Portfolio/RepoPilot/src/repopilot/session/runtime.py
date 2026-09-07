"""Persistent interactive turns built on the shared AgentKernel."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from repopilot.agents.background import BackgroundAgentManager, BackgroundAgentSnapshot
from repopilot.context.builder import ContextBuilder
from repopilot.context.manager import ContextManager, ContextUsage
from repopilot.core.contracts import (
    AgentState,
    ErrorType,
    JSONValue,
    Message,
    ModelResponse,
    PolicyOutcome,
    RunStatus,
    ToolCall,
    ToolResult,
)
from repopilot.core.events import RuntimeEvent, RuntimeEventKind
from repopilot.core.kernel import AgentKernel, AgentRuntimeConfig, EventSink
from repopilot.extensions.hooks import HookDispatcher, HookNotice
from repopilot.extensions.plugins import PluginManifest, discover_plugins, plugin_skill_roots
from repopilot.orchestration.reviewer import ReviewerTool
from repopilot.process.manager import ProcessManager, ProcessSnapshot
from repopilot.providers.base import ModelProvider, ModelProviderError
from repopilot.runtime.cancellation import CancellationToken, OperationCancelledError
from repopilot.runtime.policy import ApprovalHandler, PermissionEngine, PermissionMode
from repopilot.runtime.runner import CommandRunner
from repopilot.runtime.tool_requirements import required_tool_from_prompt
from repopilot.security.paths import PathSecurityError, resolve_workspace_path
from repopilot.session.background_tasks import BackgroundTaskRecord
from repopilot.session.exporter import ExportPreview, preview_session_export, write_session_export
from repopilot.session.models import SessionMetadata
from repopilot.session.planning import SessionPlan, TodoItem, TodoStatus
from repopilot.session.snapshots import SessionTurnSnapshot, TurnFileChange, workspace_delta
from repopilot.session.store import SessionStore
from repopilot.session.workflow import (
    TurnWorkflowReport,
    WorkflowStage,
    checks_from_tool_results,
)
from repopilot.skills.registry import Skill, SkillRegistry
from repopilot.tools.base import TodoWriter, Tool, ToolContext, ToolRegistry
from repopilot.tools.coding import SnapshotLimitError, capture_text_snapshot
from repopilot.verification.engine import (
    VerificationCommand,
    VerificationEngine,
    VerificationPlan,
    VerificationReport,
    VerificationResult,
    default_verification_commands,
    default_verification_plan,
)
from repopilot.workspace.contracts import InteractiveTask
from repopilot.workspace.project import ProjectWorkspace
from repopilot.workspace.settings import ToolPermissionRules

_SESSION_SNAPSHOT_EXCLUDE_PATHS = (
    ".mypy_cache.bak",
    ".mypy_cache_bak",
    "Reference code",
    "artifacts",
    "build",
    "evaluation",
    "external",
)
_SESSION_SNAPSHOT_MAX_FILES = 5_000
_SESSION_SNAPSHOT_MAX_TOTAL_BYTES = 64_000_000
_UNDO_MAX_FILE_BYTES = 1_000_000
_UNDO_MAX_ENTRIES = 20
_TURN_SNAPSHOT_MAX_DELTA_BYTES = 4_000_000
_EMPTY_WORKFLOW_REPORT = TurnWorkflowReport()


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SessionTurnResult:
    """The durable result of one submitted user message."""

    metadata: SessionMetadata
    answer: str
    state: AgentState
    workflow: TurnWorkflowReport = _EMPTY_WORKFLOW_REPORT


@dataclass(frozen=True, slots=True)
class SessionContextResult:
    """Persisted compaction output and the current model-context accounting."""

    metadata: SessionMetadata
    summary: str
    usage: ContextUsage


@dataclass(frozen=True, slots=True)
class SessionVerificationResult:
    """A user-triggered verification report associated with a durable session."""

    metadata: SessionMetadata
    report: VerificationReport


@dataclass(frozen=True, slots=True)
class SessionEvidence:
    """Read-only local evidence and policy facts for one interactive session."""

    metadata: SessionMetadata
    permission_mode: PermissionMode
    allowed_tools: tuple[str, ...]
    denied_tools: tuple[str, ...]
    latest_verification: VerificationReport | None
    latest_snapshot: SessionTurnSnapshot | None

    @property
    def repair_ready(self) -> bool:
        """A repair is meaningful only after a recorded failing verification."""
        return self.latest_verification is not None and not self.latest_verification.ok


@dataclass(frozen=True, slots=True)
class SessionWorkflowStatus:
    """A local-only view of the next safe step in an interactive coding workflow."""

    metadata: SessionMetadata
    plan: SessionPlan | None
    latest_verification: VerificationReport | None
    latest_snapshot: SessionTurnSnapshot | None
    next_action: str

    @property
    def repair_ready(self) -> bool:
        return self.latest_verification is not None and not self.latest_verification.ok

    def render(self) -> str:
        if self.plan is None or not self.plan.todos:
            plan_text = "none (optional: draft with /plan draft <step> | <step>)"
        else:
            completed = sum(item.status is TodoStatus.COMPLETED for item in self.plan.todos)
            state = "approved" if self.plan.approved else "draft"
            plan_text = f"{state} ({completed}/{len(self.plan.todos)} complete)"
        if self.latest_snapshot is None:
            changes_text = "no completed-turn snapshot"
        elif self.latest_snapshot.inventory_error:
            changes_text = f"unavailable ({self.latest_snapshot.inventory_error})"
        elif self.latest_snapshot.files:
            changes_text = ", ".join(change.path for change in self.latest_snapshot.files)
        else:
            changes_text = "none"
        if self.latest_verification is None:
            verification_text = "not run"
        elif self.latest_verification.ok:
            verification_text = "passed"
        else:
            verification_text = "failed (repair available)"
        return (
            "Workflow (local only; no model or command was started):\n"
            f"Plan: {plan_text}\n"
            f"Latest changed files: {changes_text}\n"
            f"Verification: {verification_text}\n"
            f"Next safe action: {self.next_action}\n"
            "Plan approval is advisory; each tool action still follows the active permission "
            "policy."
        )


@dataclass(frozen=True, slots=True)
class SessionRewindPreview:
    """A no-write description of a requested historical session revision."""

    target_sequence: int
    changed_files: tuple[str, ...]
    conflicts: tuple[str, ...]
    unavailable_sequences: tuple[int, ...]
    manual_cleanup_files: tuple[str, ...]

    @property
    def can_apply(self) -> bool:
        return (
            not self.conflicts and not self.unavailable_sequences and not self.manual_cleanup_files
        )


@dataclass(frozen=True, slots=True)
class SessionRewindResult:
    """Outcome of an explicitly confirmed code, conversation, or full rewind."""

    metadata: SessionMetadata
    target_sequence: int
    scope: str
    restored_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SessionReviewResult:
    """One explicit, read-only working-tree review with its bounded Git evidence."""

    metadata: SessionMetadata
    mode: str
    diff: str
    review: ToolResult


@dataclass(frozen=True, slots=True)
class SessionExportResult:
    """One explicitly confirmed, local-only conversation export."""

    metadata: SessionMetadata
    preview: ExportPreview


@dataclass(frozen=True, slots=True)
class SessionBackgroundTasksResult:
    """The latest durable, non-replayable receipt view for session child processes."""

    metadata: SessionMetadata
    tasks: tuple[BackgroundTaskRecord, ...]


@dataclass(frozen=True, slots=True)
class _UndoEdit:
    """One in-memory, conflict-checked reversible edit in an open session."""

    relative_path: str
    original_content: str
    after_digest: str


class SessionRuntime:
    """Run a bounded coding turn while retaining an append-only session transcript."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        tools: list[Tool],
        runner: CommandRunner,
        store: SessionStore,
        model: str,
        provider_label: str | None = None,
        permission_mode: PermissionMode = PermissionMode.MANUAL,
        permission_rules: ToolPermissionRules | None = None,
        stream_model_output: bool = False,
        approval: ApprovalHandler | None = None,
        context_builder: ContextBuilder | None = None,
        context_manager: ContextManager | None = None,
    ) -> None:
        self.store = store
        self._provider = provider
        self.model = model
        self.provider_label = provider_label or model
        self.provider_name = self.provider_label.partition("/")[0]
        self.runner = runner
        self.approval = approval
        self.permission = PermissionEngine(permission_mode, permission_rules)
        self.verification = VerificationEngine(runner)
        self.context_builder = context_builder or ContextBuilder()
        self.context_manager = context_manager or ContextManager()
        self.registry = ToolRegistry(tools)
        self._runtime_config = AgentRuntimeConfig(stream_model_output=stream_model_output)
        self.kernel = AgentKernel(
            provider=provider,
            registry=self.registry,
            policy=self.permission,
            approval=approval,
            config=self._runtime_config,
        )
        self.permission_mode = permission_mode
        self._process_managers: dict[str, ProcessManager] = {}
        self._subagent_managers: dict[str, BackgroundAgentManager] = {}
        self._hook_dispatchers: dict[str, HookDispatcher] = {}
        self._session_metadata: dict[str, SessionMetadata] = {}
        self._active_cancellations: dict[str, CancellationToken] = {}
        self._undo_stacks: dict[str, list[_UndoEdit]] = {}

    def replace_provider(
        self,
        *,
        provider: ModelProvider,
        model: str,
        tools: list[Tool],
        provider_label: str | None = None,
    ) -> None:
        """Switch the model at a prompt boundary while retaining session state.

        Model-backed reviewer and subagent tools are recreated by the CLI along
        with the kernel.  File baselines, checkpoints, permissions, process
        managers, and the transcript stay attached to the existing session.
        """
        if self._active_cancellations:
            raise RuntimeError("cannot change the model while a turn is running")
        if any(manager.has_running for manager in self._subagent_managers.values()):
            raise RuntimeError("cannot change the model while a background subagent is running")
        self._provider = provider
        self.model = model
        self.provider_label = provider_label or model
        self.provider_name = self.provider_label.partition("/")[0]
        self.registry = ToolRegistry(tools)
        self.kernel = AgentKernel(
            provider=provider,
            registry=self.registry,
            policy=self.permission,
            approval=self.approval,
            config=self._runtime_config,
        )

    def start(self, project_root: Path) -> SessionMetadata:
        project = ProjectWorkspace.discover(project_root)
        dispatcher = HookDispatcher.load(project.root)
        exclusions = self._snapshot_exclusions(project.root)
        # Build the baseline first: an oversized project must not leave a half-created session.
        baseline = capture_text_snapshot(
            project.root,
            exclude_paths=exclusions,
            max_files=_SESSION_SNAPSHOT_MAX_FILES,
            max_total_bytes=_SESSION_SNAPSHOT_MAX_TOTAL_BYTES,
        )
        metadata = self.store.create(
            project_root=project.root,
            model=self.model,
            permission_mode=self.permission_mode,
            provider=self.provider_name,
        )
        self.store.baseline_path(metadata).write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._hook_dispatchers[metadata.session_id] = dispatcher
        metadata = self._record_hook_notices(
            metadata,
            dispatcher.notices_for("session_start"),
        )
        self._session_metadata[metadata.session_id] = metadata
        return metadata

    async def run_turn(
        self,
        metadata: SessionMetadata,
        prompt: str,
        *,
        emit: EventSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> SessionTurnResult:
        """Run one turn, making cancellation durable and safe to resume from."""
        token = cancellation or CancellationToken()
        with self.store.lease(metadata):
            if metadata.session_id in self._active_cancellations:
                raise RuntimeError("session already has an active turn")
            self._active_cancellations[metadata.session_id] = token
            before, before_error = self._capture_session_workspace(metadata)
            try:
                result = await self._run_turn(metadata, prompt, emit=emit, cancellation=token)
            except OperationCancelledError as error:
                result = await self._cancel_turn(metadata, str(error), emit=emit)
            except asyncio.CancelledError:
                await self._cancel_turn(metadata, "operation cancelled", emit=emit)
                raise
            finally:
                self._active_cancellations.pop(metadata.session_id, None)
            return self._save_turn_snapshot(result, before=before, inventory_error=before_error)

    def cancel(self, metadata: SessionMetadata, *, reason: str = "operation cancelled") -> bool:
        """Request cancellation for the current turn of one session, if any."""
        token = self._active_cancellations.get(metadata.session_id)
        if token is None:
            return False
        token.cancel(reason)
        return True

    def undo_last_edit(self, metadata: SessionMetadata) -> str:
        """Restore the latest small edit when the file still matches its post-edit state."""
        stack = self._undo_stacks.get(metadata.session_id, [])
        if not stack:
            return "No reversible edit is available in this open session."
        edit = stack[-1]
        candidate = metadata.project_root / edit.relative_path
        try:
            current = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            return f"Cannot read {edit.relative_path} for undo: {error}"
        if _content_digest(current) != edit.after_digest:
            return (
                f"Cannot undo {edit.relative_path}: the file changed after RepoPilot's edit. "
                "Review the diff manually instead."
            )
        try:
            candidate.write_text(edit.original_content, encoding="utf-8", newline="")
        except OSError as error:
            return f"Cannot restore {edit.relative_path}: {error}"
        stack.pop()
        return f"Reverted RepoPilot's latest edit to {edit.relative_path}."

    def turn_snapshots(
        self, metadata: SessionMetadata, *, limit: int = 30
    ) -> tuple[SessionTurnSnapshot, ...]:
        """Return chronological, read-only workspace revisions for the active session."""

        return self.store.load_turn_snapshots(metadata, limit=limit)

    def turn_diff(
        self, metadata: SessionMetadata, sequence: int | None = None
    ) -> SessionTurnSnapshot:
        """Return one stored text delta; rendering remains a CLI responsibility."""

        snapshots = self.turn_snapshots(metadata)
        if not snapshots:
            raise ValueError("no completed turn snapshots are available")
        if sequence is None:
            return snapshots[-1]
        snapshot = next((item for item in snapshots if item.sequence == sequence), None)
        if snapshot is None:
            raise ValueError("turn snapshot is unknown; use /rewind list")
        return snapshot

    def rewind_preview(self, metadata: SessionMetadata, sequence: int) -> SessionRewindPreview:
        """Compute a no-write, conflict-aware reverse delta to an earlier turn."""

        snapshots = self.turn_snapshots(metadata, limit=10_000)
        target = next((item for item in snapshots if item.sequence == sequence), None)
        if target is None:
            raise ValueError("turn snapshot is unknown; use /rewind list")
        later = tuple(item for item in snapshots if item.sequence > sequence)
        unavailable = tuple(item.sequence for item in later if not item.rewindable)
        expected: dict[str, str | None] = {}
        desired: dict[str, str | None] = {}
        for snapshot in reversed(later):
            for change in snapshot.files:
                expected.setdefault(change.path, change.after)
                desired[change.path] = change.before
        task = InteractiveTask(
            task_id=f"session:{metadata.session_id}",
            workspace=metadata.project_root,
            problem_statement="Preview an explicit historical session rewind.",
        )
        conflicts: list[str] = []
        for path, expected_content in expected.items():
            candidate = resolve_workspace_path(task, path)
            try:
                actual = candidate.read_text(encoding="utf-8") if candidate.is_file() else None
            except (OSError, UnicodeDecodeError):
                actual = None
            if actual != expected_content:
                conflicts.append(path)
        manual_cleanup = tuple(sorted(path for path, content in desired.items() if content is None))
        return SessionRewindPreview(
            sequence,
            tuple(sorted(desired)),
            tuple(sorted(conflicts)),
            unavailable,
            manual_cleanup,
        )

    def rewind(
        self, metadata: SessionMetadata, sequence: int, *, scope: str = "all"
    ) -> SessionRewindResult:
        """Apply a user-confirmed historical revision without overwriting conflicts."""

        if scope not in {"all", "code", "session"}:
            raise ValueError("rewind scope must be all, code, or session")
        with self.store.lease(metadata):
            snapshots = self.turn_snapshots(metadata, limit=10_000)
            target = next((item for item in snapshots if item.sequence == sequence), None)
            if target is None:
                raise ValueError("turn snapshot is unknown; use /rewind list")
            preview = self.rewind_preview(metadata, sequence)
            if preview.unavailable_sequences:
                items = ", ".join(str(item) for item in preview.unavailable_sequences)
                raise ValueError(f"cannot rewind across incomplete turn snapshots: {items}")
            if scope != "session" and preview.manual_cleanup_files:
                raise ValueError(
                    "cannot rewind files that need deletion; remove them manually first: "
                    + ", ".join(preview.manual_cleanup_files)
                )
            if scope != "session" and preview.conflicts:
                raise ValueError(
                    "cannot rewind because files changed after their recorded state: "
                    + ", ".join(preview.conflicts)
                )
            before, before_error = self._capture_session_workspace(metadata)
            if before_error is not None:
                raise ValueError(f"cannot record a safe rewind snapshot: {before_error}")
            restored_files: tuple[str, ...] = ()
            if scope != "session":
                restored_files = self._apply_rewind_code(metadata, snapshots, sequence)
            state = self.store.checkpoint(metadata).load()
            if state is None:
                state = AgentState(metadata.session_id, f"session:{metadata.session_id}")
            updated = metadata
            if scope != "code":
                state = AgentState.from_dict(target.state.to_dict())
                state.pending_calls.clear()
                updated = self.store.restore_context_summary(updated, target.context_summary)
                updated = self.store.save_plan(
                    updated, target.plan if target.plan is not None else SessionPlan("", False, ())
                )
            updated = self.store.append_runtime_event(
                updated,
                kind="session_rewound",
                data={
                    "target_sequence": sequence,
                    "scope": scope,
                    "restored_files": list(restored_files),
                },
            )
            after, after_error = self._capture_session_workspace(updated)
            snapshot = self._build_turn_snapshot(
                updated,
                state,
                before=before,
                after=after,
                inventory_error=after_error,
                kind="rewind",
            )
            updated = self.store.save_turn_snapshot(updated, snapshot)
            self.store.checkpoint(updated).save(state)
            return SessionRewindResult(updated, sequence, scope, restored_files)

    async def review_working_tree(
        self,
        metadata: SessionMetadata,
        *,
        mode: str = "quality",
        emit: EventSink | None = None,
    ) -> SessionReviewResult:
        """Review only the current uncommitted Git diff; no review path can write a file."""

        if mode not in {"quality", "security"}:
            raise ValueError("review mode must be quality or security")
        with self.store.lease(metadata):
            diff_tool = self.registry.get("git_working_diff")
            if diff_tool is None or not diff_tool.spec.read_only:
                raise RuntimeError("read-only Git diff is unavailable")
            task = InteractiveTask(
                task_id=f"session:{metadata.session_id}",
                workspace=metadata.project_root,
                problem_statement="Review current uncommitted source changes.",
            )
            context = ToolContext(task, self.runner, cancellation=None)
            diff_result = await diff_tool.run(
                ToolCall("interactive-working-diff", "git_working_diff", {}), context
            )
            if not diff_result.ok:
                return SessionReviewResult(metadata, mode, "", diff_result)
            raw_output = (
                diff_result.data.get("stdout") if isinstance(diff_result.data, dict) else ""
            )
            diff = raw_output if isinstance(raw_output, str) else ""
            state = self.store.checkpoint(metadata).load()
            if state is None:
                state = AgentState(metadata.session_id, f"session:{metadata.session_id}")
            updated = await self._dispatch_event(
                metadata,
                RuntimeEvent(
                    RuntimeEventKind.REVIEW_STARTED, {"mode": mode, "diff_bytes": len(diff)}
                ),
                emit=emit,
            )
            if not diff.strip():
                review = ToolResult(
                    f"interactive-{mode}-review-{state.iteration}",
                    "review_diff",
                    True,
                    {"verdict": "clean", "issues": [], "confidence": "high"},
                )
            else:
                review = await ReviewerTool(self._provider).run(
                    ToolCall(
                        f"interactive-{mode}-review-{state.iteration}",
                        "review_diff",
                        {
                            "mode": mode,
                            "problem": "Review the current uncommitted Git diff.",
                            "diff": diff,
                        },
                    ),
                    context,
                )
            message = Message(
                "tool",
                json.dumps(review.to_dict(), ensure_ascii=False, sort_keys=True),
                name="security_review" if mode == "security" else "review",
                tool_call_id=review.call_id,
            )
            state.messages.append(message)
            updated = self.store.append_message(updated, message)
            self.store.checkpoint(updated).save(state)
            updated = await self._dispatch_event(
                updated,
                RuntimeEvent(
                    RuntimeEventKind.REVIEW_COMPLETED,
                    {"mode": mode, "ok": review.ok, "diff_bytes": len(diff)},
                ),
                emit=emit,
            )
            return SessionReviewResult(updated, mode, diff, review)

    async def background_tasks(self, metadata: SessionMetadata) -> SessionBackgroundTasksResult:
        """Refresh safe task receipts; a restarted CLI never reattaches or restarts a child."""

        manager = self._process_manager(metadata)
        known = manager.known()
        snapshots = tuple(
            await asyncio.gather(*(manager.snapshot(item.process_id) for item in known))
        )
        previous = self.store.load_background_tasks(metadata)
        records = {item.process_id: item for item in previous}
        active_ids = {snapshot.process.process_id for snapshot in snapshots}
        for snapshot in snapshots:
            record = records.get(snapshot.process.process_id)
            if record is not None:
                records[record.process_id] = record.observed(snapshot)
        for process_id, record in tuple(records.items()):
            if record.status == "running" and process_id not in active_ids:
                records[process_id] = record.unavailable_after_restart()
        refreshed = tuple(sorted(records.values(), key=lambda item: item.started_at, reverse=True))
        if refreshed != previous:
            metadata = self.store.save_background_tasks(metadata, refreshed)
            self._session_metadata[metadata.session_id] = metadata
        return SessionBackgroundTasksResult(metadata, refreshed)

    async def stop_background_task(
        self, metadata: SessionMetadata, process_id: str
    ) -> tuple[SessionMetadata, ProcessSnapshot]:
        """Stop exactly one session-owned background task after CLI confirmation."""
        stopped = await self._process_manager(metadata).terminate(process_id)
        records = {item.process_id: item for item in self.store.load_background_tasks(metadata)}
        record = records.get(process_id)
        if record is not None:
            records[process_id] = record.observed(stopped, stopped=True)
            metadata = self.store.save_background_tasks(
                metadata,
                tuple(sorted(records.values(), key=lambda item: item.started_at, reverse=True)),
            )
            self._session_metadata[metadata.session_id] = metadata
        return metadata, stopped

    async def background_task_log(
        self, metadata: SessionMetadata, process_id: str
    ) -> ProcessSnapshot:
        """Read bounded output from a live child only; output is never persisted to a receipt."""

        return await self._process_manager(metadata).snapshot(process_id)

    async def retry_background_task(
        self, metadata: SessionMetadata, process_id: str
    ) -> tuple[SessionMetadata, BackgroundTaskRecord]:
        """Start a new child only after the CLI has explicitly confirmed this retry."""

        records = {item.process_id: item for item in self.store.load_background_tasks(metadata)}
        source = records.get(process_id)
        if source is None:
            raise ValueError("background task receipt is unknown")
        if source.status == "running":
            raise ValueError("background task is still running in this CLI session")
        if not source.retryable:
            raise ValueError(
                "background task argv was redacted and cannot be retried automatically"
            )
        cwd = (metadata.project_root / source.cwd).resolve(strict=True)
        process = await self._process_manager(metadata).start(source.command, cwd=cwd)
        record = BackgroundTaskRecord.from_started_result(
            process_id=process.process_id,
            command=process.command,
            cwd=process.cwd.relative_to(metadata.project_root).as_posix() or ".",
            started_at=process.started_at.isoformat(),
        )
        records[record.process_id] = record
        metadata = self.store.save_background_tasks(
            metadata,
            tuple(sorted(records.values(), key=lambda item: item.started_at, reverse=True)),
        )
        self._session_metadata[metadata.session_id] = metadata
        return metadata, record

    async def background_subagents(
        self, metadata: SessionMetadata
    ) -> tuple[BackgroundAgentSnapshot, ...]:
        """List bounded analysis-only subagent tasks for this session."""

        return await self._subagent_manager(metadata).snapshots()

    async def stop_background_subagent(
        self, metadata: SessionMetadata, task_id: str
    ) -> BackgroundAgentSnapshot:
        """Cancel one exact analysis-only subagent task."""

        return await self._subagent_manager(metadata).stop(task_id)

    async def list_worktrees(self, metadata: SessionMetadata) -> ToolResult:
        """Run the fixed, read-only worktree inventory for an explicit slash command."""
        tool = self.registry.get("list_worktrees")
        if tool is None or not tool.spec.read_only:
            return ToolResult(
                "interactive-worktrees",
                "list_worktrees",
                False,
                error="read-only worktree inventory is unavailable",
            )
        task = InteractiveTask(
            task_id=f"session:{metadata.session_id}",
            workspace=metadata.project_root,
            problem_statement="List the current Git worktrees.",
        )
        context = ToolContext(task, self.runner, cancellation=None)
        return await tool.run(ToolCall("interactive-worktrees", tool.spec.name, {}), context)

    async def inspect_worktree(self, metadata: SessionMetadata, name: str) -> ToolResult:
        """Read safety-relevant state for one managed worktree without changing it."""

        tool = self.registry.get("inspect_worktree")
        if tool is None or not tool.spec.read_only:
            return ToolResult(
                "interactive-worktree-inspect",
                "inspect_worktree",
                False,
                error="worktree inspection is unavailable",
            )
        task = InteractiveTask(
            task_id=f"session:{metadata.session_id}",
            workspace=metadata.project_root,
            problem_statement="Inspect one managed Git worktree.",
        )
        context = ToolContext(task, self.runner, cancellation=None)
        return await tool.run(
            ToolCall("interactive-worktree-inspect", tool.spec.name, {"name": name}), context
        )

    async def aclose(self) -> None:
        """Stop all session-owned background processes before releasing the runtime."""
        managers = tuple(self._process_managers.values())
        self._process_managers.clear()
        if managers:
            await asyncio.gather(*(manager.aclose() for manager in managers))
        subagent_managers = tuple(self._subagent_managers.values())
        self._subagent_managers.clear()
        if subagent_managers:
            await asyncio.gather(*(manager.aclose() for manager in subagent_managers))
        for session_id, dispatcher in tuple(self._hook_dispatchers.items()):
            metadata = self._session_metadata.get(session_id)
            if metadata is not None:
                self._session_metadata[session_id] = self._record_hook_notices(
                    metadata,
                    dispatcher.notices_for("session_stop"),
                )
        self._hook_dispatchers.clear()
        self._session_metadata.clear()

    async def _run_turn(
        self,
        metadata: SessionMetadata,
        prompt: str,
        *,
        emit: EventSink | None = None,
        cancellation: CancellationToken,
    ) -> SessionTurnResult:
        if not prompt.strip():
            raise ValueError("interactive prompt must not be empty")
        state = self.store.checkpoint(metadata).load()
        if state is None:
            state = AgentState(metadata.session_id, f"session:{metadata.session_id}")
        state.status = RunStatus.RUNNING
        user_message = Message("user", prompt)
        state.messages.append(user_message)
        metadata = self.store.append_message(metadata, user_message)
        checkpoint = self.store.checkpoint(metadata)
        checkpoint.save(state)
        journal = self.store.journal(metadata)
        baseline = self._load_baseline(metadata)
        project = ProjectWorkspace.discover(
            metadata.project_root,
            user_instructions_root=self.store.root,
        )
        verification_commands = default_verification_commands(project)
        task = InteractiveTask(
            task_id=f"session:{metadata.session_id}",
            workspace=metadata.project_root,
            problem_statement=prompt,
            # Supply a project-discovered immutable command to the dedicated test tool.
            # It uses RepoPilot's interpreter, rather than the caller's global Python.
            test_command=verification_commands[0].command if verification_commands else (),
        )
        context = ToolContext(
            task,
            self.runner,
            baseline,
            process_manager=self._process_manager(metadata),
            subagent_manager=self._subagent_manager(metadata),
            cancellation=cancellation,
            snapshot_exclude_paths=self._snapshot_exclusions(metadata.project_root),
            snapshot_max_files=_SESSION_SNAPSHOT_MAX_FILES,
            snapshot_max_total_bytes=_SESSION_SNAPSHOT_MAX_TOTAL_BYTES,
        )
        turn_baseline, turn_snapshot_error = self._workflow_snapshot(task, context)
        turn_results: list[ToolResult] = []

        def write_todos(raw_todos: list[dict[str, JSONValue]]) -> dict[str, JSONValue]:
            """Persist only structured session todos; no workspace authority is exposed."""

            nonlocal metadata
            previous = self.store.load_plan(metadata) or SessionPlan("", False, ())
            plan = SessionPlan.from_tool_items(raw_todos, previous=previous)
            metadata = self.store.save_plan(metadata, plan)
            self._sync_state_plan(state, plan)
            checkpoint.save(state)
            return plan.to_dict()

        context.todo_writer = cast(TodoWriter, write_todos)

        def dispatch(event: RuntimeEvent) -> Awaitable[None]:
            async def persist_and_forward() -> None:
                nonlocal metadata
                metadata = await self._dispatch_event(metadata, event, emit=emit)

            return persist_and_forward()

        async def transition(stage: WorkflowStage, reason: str, *, force: bool = False) -> None:
            """Persist advisory workflow milestones without granting any permissions."""

            nonlocal metadata
            if not force and state.workflow_stage == stage.value:
                return
            state.workflow_stage = stage.value
            checkpoint.save(state)
            metadata = await self._dispatch_event(
                metadata,
                RuntimeEvent(
                    RuntimeEventKind.WORKFLOW_STAGE_CHANGED,
                    {"stage": stage.value, "reason": reason},
                ),
                emit=emit,
            )

        started_iteration = state.iteration
        started_tools = state.tool_calls
        started_tokens = state.input_tokens + state.output_tokens
        required_tool = required_tool_from_prompt(
            prompt, (spec.name for spec in self.registry.specs())
        )
        tool_requirement_corrections = 0
        patch_recovery_corrections = 0
        patch_recovery_pending = False
        await transition(WorkflowStage.EXPLORE, "turn started", force=True)
        while True:
            cancellation.raise_if_cancelled()
            limit = self._turn_limit(
                state,
                started_iteration=started_iteration,
                started_tools=started_tools,
                started_tokens=started_tokens,
            )
            if limit is not None:
                return self._finish(metadata, state, f"Stopped: {limit}")
            if state.pending_calls:
                undo_candidates = self._capture_undo_candidates(
                    tuple(state.pending_calls), task=task
                )
                results = await self.kernel.execute_calls(
                    tuple(state.pending_calls),
                    task=task,
                    context=context,
                    state=state,
                    journal=journal,
                    persist=lambda: checkpoint.save(state),
                    emit=dispatch,
                )
                cancellation.raise_if_cancelled()
                state.pending_calls.clear()
                for result in results:
                    result = self.store.persist_large_tool_result(metadata, result)
                    turn_results.append(result)
                    self._record_undo_if_applied(metadata, result, undo_candidates)
                    context.recent_results.append(result)
                    message = Message(
                        "tool",
                        json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True),
                        name=result.tool_name,
                        tool_call_id=result.call_id,
                    )
                    state.messages.append(message)
                    metadata = self.store.append_message(metadata, message)
                metadata = self._record_background_starts(metadata, results)
                checkpoint.save(state)
                if patch_recovery_pending and any(
                    result.tool_name == "read_file" and result.ok for result in results
                ):
                    patch_recovery_pending = False
                    await transition(
                        WorkflowStage.EXPLORE,
                        "current target file was re-read after patch conflict",
                    )
                if self._has_recoverable_patch_conflict(results):
                    patch_recovery_pending = True
                    await transition(
                        WorkflowStage.EXPLORE,
                        "recoverable patch conflict requires fresh inspection",
                    )
                if any(result.tool_name == "run_tests" for result in results):
                    await transition(WorkflowStage.VERIFY, "run_tests completed")
                elif any(result.tool_name == "write_todos" for result in results):
                    await transition(WorkflowStage.PLAN, "todo plan recorded")
                elif any(result.side_effect and result.ok for result in results):
                    await transition(WorkflowStage.IMPLEMENT, "workspace action completed")
                continue
            state.iteration += 1
            projection = self.context_manager.project(
                state.messages, summary=self.store.load_context_summary(metadata)
            )
            request = self.context_builder.build(
                task=task,
                state=state,
                tools=self.registry.specs(),
                session_summary=projection.summary,
                context_messages=projection.messages,
                project_instructions=project.instructions.render(),
                skills=self._selected_skills(metadata, prompt),
                runtime_identity=self.provider_label,
                session_memory=self.store.load_project_memory(metadata.project_root),
                required_tool=required_tool,
            )
            try:
                response = await self.kernel.request_model(
                    request,
                    iteration=state.iteration,
                    emit=dispatch,
                    cancellation=cancellation,
                )
            except ModelProviderError as error:
                return self._finish(metadata, state, f"Model provider failed: {error}")
            state.input_tokens += response.usage.input_tokens
            state.output_tokens += response.usage.output_tokens
            if patch_recovery_pending and not self._is_patch_recovery_read(response):
                if patch_recovery_corrections >= 1:
                    return self._finish(
                        metadata,
                        state,
                        (
                            "Stopped: apply_patch conflicted and the model did not re-read the "
                            "target before retrying. No additional patch was executed."
                        ),
                    )
                patch_recovery_corrections += 1
                correction = Message(
                    "user",
                    (
                        "[TRUSTED RUNTIME CORRECTION] A recoverable apply_patch conflict just "
                        "occurred. Do not make another patch yet. Call read_file for the current "
                        "target file by itself, use its recorded content, then propose one minimal "
                        "new patch. If reading fails, report the actual blocker."
                    ),
                )
                state.messages.append(correction)
                metadata = self.store.append_message(metadata, correction)
                checkpoint.save(state)
                await self._dispatch_event(
                    metadata,
                    RuntimeEvent(
                        RuntimeEventKind.PATCH_RECOVERY_ENFORCED,
                        {"correction": patch_recovery_corrections, "required_tool": "read_file"},
                    ),
                    emit=emit,
                )
                continue
            if required_tool is not None and required_tool not in {
                call.name for call in response.tool_calls
            }:
                if tool_requirement_corrections >= 1:
                    return self._finish(
                        metadata,
                        state,
                        (
                            "Stopped: the configured model did not call the explicitly required "
                            f"tool {required_tool!r}. No unproven tool result was accepted."
                        ),
                    )
                tool_requirement_corrections += 1
                correction = Message(
                    "user",
                    (
                        "[TRUSTED RUNTIME CORRECTION] Do not provide a final answer yet. "
                        f"Call the required structured tool {required_tool!r} now. If its "
                        "execution is denied or fails, report that actual result afterwards."
                    ),
                )
                state.messages.append(correction)
                metadata = self.store.append_message(metadata, correction)
                checkpoint.save(state)
                await self._dispatch_event(
                    metadata,
                    RuntimeEvent(
                        RuntimeEventKind.TOOL_REQUIREMENT_ENFORCED,
                        {
                            "tool": required_tool,
                            "correction": tool_requirement_corrections,
                        },
                    ),
                    emit=emit,
                )
                continue
            if response.tool_calls:
                requested = Message(
                    "assistant",
                    response.content,
                    tool_calls=response.tool_calls,
                    reasoning_content=response.reasoning_content,
                )
                state.messages.append(requested)
                metadata = self.store.append_message(metadata, requested)
                state.pending_calls = list(response.tool_calls)
                required_tool = None
                checkpoint.save(state)
                continue
            report = self._workflow_report(
                task,
                context,
                before=turn_baseline,
                snapshot_error=turn_snapshot_error,
                tool_results=turn_results,
            )
            await transition(WorkflowStage.REVIEW, "model supplied a final answer")
            if report.has_evidence:
                metadata = await self._dispatch_event(
                    metadata,
                    RuntimeEvent(RuntimeEventKind.WORKFLOW_REPORT_CREATED, report.to_dict()),
                    emit=emit,
                )
            await transition(WorkflowStage.ANSWER, "workflow evidence finalized")
            return self._finish(metadata, state, self._content(response), workflow=report)

    @staticmethod
    def _has_recoverable_patch_conflict(results: tuple[ToolResult, ...]) -> bool:
        """Identify the only patch failure that benefits from reading fresh contents."""

        return any(
            result.tool_name == "apply_patch"
            and not result.ok
            and result.recoverable
            and result.error_type is ErrorType.CONFLICT
            for result in results
        )

    @staticmethod
    def _is_patch_recovery_read(response: ModelResponse) -> bool:
        """Require a standalone workspace read before the model retries a conflicted edit."""

        names = {call.name for call in response.tool_calls}
        return names == {"read_file"}

    @staticmethod
    def _capture_undo_candidates(
        calls: tuple[ToolCall, ...], *, task: InteractiveTask
    ) -> dict[str, tuple[str, str]]:
        """Capture pre-edit contents without trusting an unvalidated model path."""
        candidates: dict[str, tuple[str, str]] = {}
        for call in calls:
            if call.name != "apply_patch":
                continue
            raw_path = call.arguments.get("path")
            if not isinstance(raw_path, str):
                continue
            try:
                path = resolve_workspace_path(task, raw_path)
                if not path.is_file() or path.stat().st_size > _UNDO_MAX_FILE_BYTES:
                    continue
                candidates[call.call_id] = (
                    path.relative_to(task.workspace).as_posix(),
                    path.read_text(encoding="utf-8"),
                )
            except (OSError, UnicodeDecodeError, PathSecurityError):
                continue
        return candidates

    def _record_undo_if_applied(
        self,
        metadata: SessionMetadata,
        result: ToolResult,
        candidates: dict[str, tuple[str, str]],
    ) -> None:
        candidate = candidates.get(result.call_id)
        if result.tool_name != "apply_patch" or not result.ok or candidate is None:
            return
        relative_path, original_content = candidate
        path = metadata.project_root / relative_path
        try:
            current = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        stack = self._undo_stacks.setdefault(metadata.session_id, [])
        stack.append(_UndoEdit(relative_path, original_content, _content_digest(current)))
        if len(stack) > _UNDO_MAX_ENTRIES:
            del stack[:-_UNDO_MAX_ENTRIES]

    def _capture_session_workspace(
        self, metadata: SessionMetadata
    ) -> tuple[dict[str, str], str | None]:
        """Take the bounded text inventory needed for durable turn-level rewind."""

        try:
            return (
                capture_text_snapshot(
                    metadata.project_root,
                    exclude_paths=self._snapshot_exclusions(metadata.project_root),
                    max_files=_SESSION_SNAPSHOT_MAX_FILES,
                    max_total_bytes=_SESSION_SNAPSHOT_MAX_TOTAL_BYTES,
                ),
                None,
            )
        except (OSError, UnicodeDecodeError, ValueError, SnapshotLimitError) as error:
            return {}, str(error)

    def _build_turn_snapshot(
        self,
        metadata: SessionMetadata,
        state: AgentState,
        *,
        before: dict[str, str],
        after: dict[str, str],
        inventory_error: str | None,
        kind: str = "turn",
    ) -> SessionTurnSnapshot:
        """Build a bounded immutable revision; failed inventory stays visible but non-rewindable."""

        delta: tuple[TurnFileChange, ...] = ()
        error = inventory_error
        if error is None:
            delta, error = workspace_delta(before, after, max_bytes=_TURN_SNAPSHOT_MAX_DELTA_BYTES)
        return SessionTurnSnapshot(
            self.store.next_turn_snapshot_sequence(metadata),
            AgentState.from_dict(state.to_dict()),
            delta,
            self.store.load_context_summary(metadata),
            self.store.load_plan(metadata),
            kind=kind,
            inventory_error=error,
        )

    def _save_turn_snapshot(
        self,
        result: SessionTurnResult,
        *,
        before: dict[str, str],
        inventory_error: str | None,
    ) -> SessionTurnResult:
        """Persist a revision after every returned turn, including cancellations and failures."""

        after, after_error = self._capture_session_workspace(result.metadata)
        snapshot = self._build_turn_snapshot(
            result.metadata,
            result.state,
            before=before,
            after=after,
            inventory_error=inventory_error or after_error,
        )
        updated = self.store.save_turn_snapshot(result.metadata, snapshot)
        self.store.checkpoint(updated).save(result.state)
        return SessionTurnResult(updated, result.answer, result.state, result.workflow)

    def _apply_rewind_code(
        self,
        metadata: SessionMetadata,
        snapshots: tuple[SessionTurnSnapshot, ...],
        sequence: int,
    ) -> tuple[str, ...]:
        """Reverse text replacements after preview has ruled out conflicts and deletions."""

        desired: dict[str, str | None] = {}
        for snapshot in reversed(tuple(item for item in snapshots if item.sequence > sequence)):
            for change in snapshot.files:
                desired[change.path] = change.before
        if any(content is None for content in desired.values()):
            raise ValueError("this rewind requires a manual file deletion")
        task = InteractiveTask(
            task_id=f"session:{metadata.session_id}",
            workspace=metadata.project_root,
            problem_statement="Apply an explicitly confirmed historical session rewind.",
        )
        restored: list[str] = []
        for relative, content in sorted(desired.items()):
            if content is None:
                continue
            path = resolve_workspace_path(task, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="")
            restored.append(relative)
        return tuple(restored)

    async def _cancel_turn(
        self,
        metadata: SessionMetadata,
        reason: str,
        *,
        emit: EventSink | None,
    ) -> SessionTurnResult:
        """Persist an interrupted turn without replaying a pending side effect on resume."""
        state = self.store.checkpoint(metadata).load()
        if state is None:
            state = AgentState(metadata.session_id, f"session:{metadata.session_id}")
        state.pending_calls.clear()
        state.status = RunStatus.CANCELLED
        answer = f"Cancelled: {reason}"
        state.final_answer = answer
        state.messages.append(Message("assistant", answer))
        updated = self.store.append_message(metadata, state.messages[-1])
        updated = await self._dispatch_event(
            updated,
            RuntimeEvent(RuntimeEventKind.TURN_CANCELLED, {"reason": reason}),
            emit=emit,
        )
        self.store.checkpoint(updated).save(state)
        return SessionTurnResult(updated, answer, state)

    def compact(self, metadata: SessionMetadata) -> SessionContextResult:
        """Persist a deterministic handoff summary while retaining the full transcript."""
        with self.store.lease(metadata):
            state = self.store.checkpoint(metadata).load()
            messages = state.messages if state is not None else []
            summary = self.context_manager.compact(
                messages, previous_summary=self.store.load_context_summary(metadata)
            )
            updated = self.store.save_context_summary(metadata, summary)
            projection = self.context_manager.project(messages, summary=summary)
            return SessionContextResult(updated, summary, projection.usage)

    def context_usage(self, metadata: SessionMetadata) -> ContextUsage:
        """Report what the next model call would retain, without mutating the session."""
        state = self.store.checkpoint(metadata).load()
        messages = state.messages if state is not None else []
        return self.context_manager.project(
            messages, summary=self.store.load_context_summary(metadata)
        ).usage

    def export_preview(
        self, metadata: SessionMetadata, *, filename: str | None = None
    ) -> ExportPreview:
        """Render a redacted local export without changing session or workspace state."""
        state = self.store.checkpoint(metadata).load()
        messages = state.messages if state is not None else []
        return preview_session_export(
            metadata,
            messages,
            export_directory=self.store.session_dir(metadata.project_root, metadata.session_id)
            / "exports",
            filename=filename,
        )

    def export_session(
        self, metadata: SessionMetadata, *, filename: str | None = None
    ) -> SessionExportResult:
        """Create a user-confirmed export, refusing to overwrite an existing file."""
        with self.store.lease(metadata):
            preview = self.export_preview(metadata, filename=filename)
            target = write_session_export(preview)
            updated = self.store.append_runtime_event(
                metadata,
                kind="session_exported",
                data={
                    "filename": target.name,
                    "message_count": preview.message_count,
                    "byte_count": preview.byte_count,
                },
            )
        return SessionExportResult(updated, preview)

    def statistics(self, metadata: SessionMetadata) -> dict[str, object]:
        """Return local, credential-free facts about the active durable session."""

        state = self.store.checkpoint(metadata).load()
        events = self.store.runtime_events(metadata)
        models = [
            event
            for event in events
            if event.get("kind") == RuntimeEventKind.MODEL_CALL_COMPLETED.value
        ]
        tools = [
            event
            for event in events
            if event.get("kind") == RuntimeEventKind.TOOL_CALL_COMPLETED.value
        ]
        retries = [
            event for event in events if event.get("kind") == RuntimeEventKind.MODEL_RETRYING.value
        ]
        requirements = [
            event
            for event in events
            if event.get("kind") == RuntimeEventKind.TOOL_REQUIREMENT_ENFORCED.value
        ]
        durations = [
            event.get("data", {}).get("duration_ms")
            for event in models
            if isinstance(event.get("data"), dict)
        ]
        model_durations = [value for value in durations if isinstance(value, int)]
        usage = self.context_usage(metadata)
        return {
            "session_id": metadata.session_id,
            "provider": self.provider_name,
            "model": self.model,
            "turns": len(models),
            "tool_calls": state.tool_calls if state is not None else len(tools),
            "input_tokens": state.input_tokens if state is not None else 0,
            "output_tokens": state.output_tokens if state is not None else 0,
            "model_retries": len(retries),
            "tool_requirement_corrections": len(requirements),
            "model_latency_ms_total": sum(model_durations),
            "model_latency_ms_latest": model_durations[-1] if model_durations else None,
            "context_messages": usage.retained_messages,
            "context_characters": usage.retained_characters,
        }

    def trace(self, metadata: SessionMetadata, *, limit: int = 30) -> tuple[dict[str, object], ...]:
        """Expose concise evidence events without rendering transcript messages or secrets."""

        raw_events = self.store.runtime_events(metadata, limit=limit)
        result: list[dict[str, object]] = []
        for event in raw_events:
            kind = event.get("kind")
            data = event.get("data")
            if not isinstance(kind, str) or not isinstance(data, dict):
                continue
            result.append({"kind": kind, "data": data})
        return tuple(result)

    def available_skills(self, metadata: SessionMetadata) -> tuple[tuple[str, str], ...]:
        """List metadata only; instructions remain deferred until selection."""
        return self._skill_registry(metadata).descriptors()

    def plan(self, metadata: SessionMetadata) -> SessionPlan | None:
        """Return the durable plan without changing its approval state."""

        return self.store.load_plan(metadata)

    def verification_plan(self, metadata: SessionMetadata) -> VerificationPlan:
        """Discover selectable checks for this project without executing them."""

        return default_verification_plan(ProjectWorkspace.discover(metadata.project_root))

    def verification_history(
        self, metadata: SessionMetadata, *, limit: int = 10
    ) -> tuple[VerificationReport, ...]:
        """Return persisted newest-first reports without exposing unredacted logs."""

        return self.store.load_verification_reports(metadata, limit=limit)

    def evidence(self, metadata: SessionMetadata) -> SessionEvidence:
        """Return persisted evidence without starting a command or contacting a model."""

        verification = self.verification_history(metadata, limit=1)
        snapshots = self.turn_snapshots(metadata, limit=1)
        rules = self.permission.rules
        return SessionEvidence(
            metadata=metadata,
            permission_mode=self.permission_mode,
            allowed_tools=rules.allow,
            denied_tools=rules.deny,
            latest_verification=verification[0] if verification else None,
            latest_snapshot=snapshots[-1] if snapshots else None,
        )

    def workflow_status(self, metadata: SessionMetadata) -> SessionWorkflowStatus:
        """Summarize plan, evidence and repair readiness without side effects."""

        plan = self.plan(metadata)
        verification = self.verification_history(metadata, limit=1)
        snapshots = self.turn_snapshots(metadata, limit=1)
        latest_verification = verification[0] if verification else None
        latest_snapshot = snapshots[-1] if snapshots else None
        if latest_verification is not None and not latest_verification.ok:
            next_action = "Review /evidence, then use /repair when you want a governed repair turn."
        elif plan is not None and plan.todos and not plan.approved:
            next_action = "Review the draft with /plan, then use /plan approve or /plan clear."
        elif latest_snapshot is not None and latest_verification is None:
            next_action = "Select a local check with /verify list, then run /verify <selector>."
        elif latest_verification is not None and latest_verification.ok:
            next_action = (
                "Verification passed; inspect /diff if you want to review changes, or continue "
                "the task."
            )
        else:
            next_action = "Submit a request, or create an optional explicit plan with /plan draft."
        return SessionWorkflowStatus(
            metadata,
            plan,
            latest_verification,
            latest_snapshot,
            next_action,
        )

    def draft_plan(
        self, metadata: SessionMetadata, steps: tuple[str, ...], *, summary: str = ""
    ) -> SessionMetadata:
        """Save a user-visible draft; edits still retain normal tool approvals."""

        if not steps:
            raise ValueError("a draft plan needs at least one non-empty step")
        todos = tuple(TodoItem(step, TodoStatus.PENDING, step) for step in steps)
        plan = SessionPlan(summary or "Session implementation plan", False, todos)
        with self.store.lease(metadata):
            updated = self.store.save_plan(metadata, plan)
            state = self.store.checkpoint(updated).load()
            if state is None:
                state = AgentState(updated.session_id, f"session:{updated.session_id}")
            self._sync_state_plan(state, plan)
            self.store.checkpoint(updated).save(state)
            return updated

    def approve_plan(self, metadata: SessionMetadata) -> SessionMetadata:
        """Record explicit user approval of the current plan without broadening permissions."""

        with self.store.lease(metadata):
            plan = self.store.load_plan(metadata)
            if plan is None or not plan.todos:
                raise ValueError("there is no non-empty plan to approve")
            approved = SessionPlan(plan.summary, True, plan.todos)
            updated = self.store.save_plan(metadata, approved)
            state = self.store.checkpoint(updated).load()
            if state is None:
                state = AgentState(updated.session_id, f"session:{updated.session_id}")
            self._sync_state_plan(state, approved)
            self.store.checkpoint(updated).save(state)
            return updated

    def clear_plan(self, metadata: SessionMetadata) -> SessionMetadata:
        """Clear plan content by overwriting one plan file; no directory removal occurs."""

        with self.store.lease(metadata):
            updated = self.store.save_plan(metadata, SessionPlan("", False, ()))
            state = self.store.checkpoint(updated).load()
            if state is not None:
                self._sync_state_plan(state, SessionPlan("", False, ()))
                self.store.checkpoint(updated).save(state)
            return updated

    def add_todo(self, metadata: SessionMetadata, content: str) -> SessionMetadata:
        """Append one user-authored pending todo without replacing the whole plan."""

        with self.store.lease(metadata):
            previous = self.store.load_plan(metadata) or SessionPlan("", False, ())
            updated_plan = SessionPlan(
                previous.summary or "Session implementation plan",
                previous.approved,
                (*previous.todos, TodoItem(content, TodoStatus.PENDING, content)),
            )
            updated = self.store.save_plan(metadata, updated_plan)
            state = self.store.checkpoint(updated).load()
            if state is not None:
                self._sync_state_plan(state, updated_plan)
                self.store.checkpoint(updated).save(state)
            return updated

    def set_todo_status(
        self, metadata: SessionMetadata, index: int, status: TodoStatus
    ) -> SessionMetadata:
        """Change one numbered todo status after validating the active plan."""

        with self.store.lease(metadata):
            previous = self.store.load_plan(metadata)
            if previous is None or not 1 <= index <= len(previous.todos):
                raise ValueError("todo number is unknown; use /todo to inspect the plan")
            todos = list(previous.todos)
            current = todos[index - 1]
            todos[index - 1] = TodoItem(current.content, status, current.active_form)
            updated_plan = SessionPlan(previous.summary, previous.approved, tuple(todos))
            updated = self.store.save_plan(metadata, updated_plan)
            state = self.store.checkpoint(updated).load()
            if state is not None:
                self._sync_state_plan(state, updated_plan)
                self.store.checkpoint(updated).save(state)
            return updated

    def available_skill_details(self, metadata: SessionMetadata) -> tuple[Skill, ...]:
        """Expose source-labelled skill metadata for interactive inspection only."""
        registry = self._skill_registry(metadata)
        return tuple(
            skill for name, _description in registry.descriptors() if (skill := registry.get(name))
        )

    def available_plugins(self, metadata: SessionMetadata) -> tuple[PluginManifest, ...]:
        """List manifest-only plugins; discovery never executes plugin code."""

        return discover_plugins(metadata.project_root, self.store.root)

    def hook_notices(self, metadata: SessionMetadata) -> tuple[HookNotice, ...]:
        """List declarative hooks; unlike shell hooks, these cannot execute code."""
        return self._hook_dispatcher(metadata).notices

    def activate_skills(self, metadata: SessionMetadata, names: tuple[str, ...]) -> SessionMetadata:
        """Persist a user-selected subset of the current project's skills."""
        with self.store.lease(metadata):
            registry = self._skill_registry(metadata)
            unknown = [name for name in names if registry.get(name) is None]
            if unknown:
                raise ValueError(f"unknown user or project skills: {', '.join(unknown)}")
            return self.store.save_active_skills(metadata, tuple(dict.fromkeys(names)))

    async def verify(
        self,
        metadata: SessionMetadata,
        commands: tuple[VerificationCommand, ...],
        *,
        emit: EventSink | None = None,
    ) -> SessionVerificationResult:
        with self.store.lease(metadata):
            return await self._verify(metadata, commands, emit=emit)

    async def _verify(
        self,
        metadata: SessionMetadata,
        commands: tuple[VerificationCommand, ...],
        *,
        emit: EventSink | None = None,
    ) -> SessionVerificationResult:
        """Run user-selected verification only after the normal execution approval."""
        if not commands:
            raise ValueError("at least one verification command is required")
        task = InteractiveTask(
            task_id=f"session:{metadata.session_id}",
            workspace=metadata.project_root,
            problem_statement="Run user-requested verification.",
        )
        shell_tool = self.registry.get("run_shell")
        if shell_tool is None:
            raise ValueError("verification requires the run_shell tool")
        for index, verification in enumerate(commands, start=1):
            call = ToolCall(
                f"verification:{index}", "run_shell", {"command": list(verification.command)}
            )
            decision = self.permission.decide(call, shell_tool.spec, task)
            if decision.outcome is PolicyOutcome.DENY:
                report = VerificationReport(
                    (self._denied_verification(verification, decision.reason),)
                )
                return self._persist_verification(metadata, report)
            if decision.outcome is PolicyOutcome.REQUIRE_APPROVAL and (
                self.approval is None or not await self.approval.approve(call, decision.reason)
            ):
                report = VerificationReport(
                    (self._denied_verification(verification, "verification was not approved"),)
                )
                return self._persist_verification(metadata, report)
        updated = await self._dispatch_event(
            metadata,
            RuntimeEvent(
                RuntimeEventKind.VERIFICATION_STARTED,
                {"commands": [list(command.command) for command in commands]},
            ),
            emit=emit,
        )
        report = await self.verification.run(metadata.project_root, commands)
        updated = await self._dispatch_event(
            updated,
            RuntimeEvent(RuntimeEventKind.VERIFICATION_COMPLETED, report.to_dict()),
            emit=emit,
        )
        return self._persist_verification(updated, report)

    async def repair(
        self, metadata: SessionMetadata, *, emit: EventSink | None = None
    ) -> SessionTurnResult:
        """Ask the main agent to repair only after a persisted failed verification report."""
        state = self.store.checkpoint(metadata).load()
        if state is None:
            raise ValueError("session has no verification history")
        reports = self.verification_history(metadata, limit=1)
        if not reports:
            raise ValueError("run /verify before requesting a repair")
        if reports[0].ok:
            raise ValueError("the latest verification passed; no repair is needed")
        return await self.run_turn(
            metadata,
            (
                "Repair the latest persisted failed verification. Inspect the affected code first, "
                "make the smallest safe change, and report what still requires verification. "
                "Do not claim verification passed unless a tool result records it. "
                "Use /evidence locally to inspect the persisted verification summary; it is not "
                "automatically forwarded to the model."
            ),
            emit=emit,
        )

    @staticmethod
    def _denied_verification(verification: VerificationCommand, reason: str) -> VerificationResult:
        return VerificationResult(verification, None, reason)

    def _persist_verification(
        self, metadata: SessionMetadata, report: VerificationReport
    ) -> SessionVerificationResult:
        updated = self.store.save_verification_report(metadata, report)
        state = self.store.checkpoint(updated).load()
        if state is None:
            state = AgentState(updated.session_id, f"session:{updated.session_id}")
        # Keep complete, redacted diagnostics in the local report file, but do
        # not automatically replay stdout/stderr to a cloud provider on a later
        # coding turn. The agent receives only a structural receipt; the user
        # can inspect the local summary through /evidence or /verify last.
        receipt = {
            "ok": report.ok,
            "results": [
                {
                    "label": item.verification.label,
                    "kind": item.verification.kind.value,
                    "ok": item.ok,
                    "exit_code": item.execution.exit_code if item.execution is not None else None,
                    "timed_out": item.execution.timed_out if item.execution is not None else False,
                }
                for item in report.results
            ],
        }
        message = Message(
            "tool",
            json.dumps(receipt, ensure_ascii=False, sort_keys=True),
            name="verification",
            tool_call_id=f"verification:{state.iteration}",
        )
        state.messages.append(message)
        updated = self.store.append_message(updated, message)
        self.store.checkpoint(updated).save(state)
        return SessionVerificationResult(updated, report)

    def _finish(
        self,
        metadata: SessionMetadata,
        state: AgentState,
        answer: str,
        *,
        workflow: TurnWorkflowReport = _EMPTY_WORKFLOW_REPORT,
    ) -> SessionTurnResult:
        message = Message("assistant", answer)
        state.final_answer = answer
        state.messages.append(message)
        state.status = RunStatus.COMPLETED
        metadata = self.store.append_message(metadata, message)
        self.store.checkpoint(metadata).save(state)
        return SessionTurnResult(metadata, answer, state, workflow)

    @staticmethod
    def _workflow_snapshot(
        task: InteractiveTask, context: ToolContext
    ) -> tuple[dict[str, str], str | None]:
        try:
            return (
                capture_text_snapshot(
                    task.workspace,
                    include_paths=task.allowed_paths,
                    exclude_paths=context.snapshot_exclude_paths,
                    max_files=context.snapshot_max_files,
                    max_total_bytes=context.snapshot_max_total_bytes,
                ),
                None,
            )
        except (OSError, SnapshotLimitError, ValueError) as error:
            return {}, str(error)

    def _workflow_report(
        self,
        task: InteractiveTask,
        context: ToolContext,
        *,
        before: dict[str, str],
        snapshot_error: str | None,
        tool_results: list[ToolResult],
    ) -> TurnWorkflowReport:
        if snapshot_error is not None:
            return TurnWorkflowReport(
                checks=checks_from_tool_results(tool_results), snapshot_error=snapshot_error
            )
        after, after_error = self._workflow_snapshot(task, context)
        if after_error is not None:
            return TurnWorkflowReport(
                checks=checks_from_tool_results(tool_results), snapshot_error=after_error
            )
        changed = tuple(
            sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        )
        return TurnWorkflowReport(changed, checks_from_tool_results(tool_results))

    def _load_baseline(self, metadata: SessionMetadata) -> dict[str, str]:
        path = self.store.baseline_path(metadata)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("session baseline must be an object")
        return {str(key): str(value) for key, value in raw.items()}

    @staticmethod
    def _snapshot_exclusions(project_root: Path) -> tuple[str, ...]:
        return tuple(
            relative
            for relative in _SESSION_SNAPSHOT_EXCLUDE_PATHS
            if (project_root / relative).is_dir()
        )

    def _process_manager(self, metadata: SessionMetadata) -> ProcessManager:
        manager = self._process_managers.get(metadata.session_id)
        if manager is None:
            manager = ProcessManager(metadata.project_root)
            self._process_managers[metadata.session_id] = manager
        return manager

    def _record_background_starts(
        self, metadata: SessionMetadata, results: tuple[ToolResult, ...]
    ) -> SessionMetadata:
        """Persist only redacted receipts after a governed start-background tool succeeds."""

        records = {item.process_id: item for item in self.store.load_background_tasks(metadata)}
        changed = False
        for result in results:
            if result.tool_name != "start_background" or not result.ok:
                continue
            data = result.data
            if not isinstance(data, dict):
                continue
            process_id = data.get("process_id")
            raw_command = data.get("command")
            cwd = data.get("cwd")
            started_at = data.get("started_at")
            if (
                not isinstance(process_id, str)
                or not isinstance(raw_command, list)
                or not all(isinstance(item, str) for item in raw_command)
                or not isinstance(cwd, str)
                or not isinstance(started_at, str)
            ):
                continue
            record = BackgroundTaskRecord.from_started_result(
                process_id=process_id,
                command=tuple(raw_command),
                cwd=cwd,
                started_at=started_at,
            )
            records[record.process_id] = record
            changed = True
        if not changed:
            return metadata
        updated = self.store.save_background_tasks(
            metadata,
            tuple(sorted(records.values(), key=lambda item: item.started_at, reverse=True)),
        )
        self._session_metadata[updated.session_id] = updated
        return updated

    def _subagent_manager(self, metadata: SessionMetadata) -> BackgroundAgentManager:
        manager = self._subagent_managers.get(metadata.session_id)
        if manager is None:
            manager = BackgroundAgentManager(self._provider)
            self._subagent_managers[metadata.session_id] = manager
        return manager

    def _hook_dispatcher(self, metadata: SessionMetadata) -> HookDispatcher:
        dispatcher = self._hook_dispatchers.get(metadata.session_id)
        if dispatcher is None:
            dispatcher = HookDispatcher.load(metadata.project_root)
            self._hook_dispatchers[metadata.session_id] = dispatcher
        return dispatcher

    def _record_hook_notices(
        self,
        metadata: SessionMetadata,
        notices: tuple[HookNotice, ...],
    ) -> SessionMetadata:
        updated = metadata
        for notice in notices:
            updated = self.store.append_runtime_event(
                updated,
                kind=RuntimeEventKind.HOOK_TRIGGERED.value,
                data={"event": notice.event, "message": notice.message},
            )
        return updated

    async def _dispatch_event(
        self,
        metadata: SessionMetadata,
        event: RuntimeEvent,
        *,
        emit: EventSink | None,
    ) -> SessionMetadata:
        """Persist one event, expose declarative hook notices, then notify the CLI."""
        updated = self.store.append_runtime_event(metadata, kind=event.kind.value, data=event.data)
        hook_events: list[RuntimeEvent] = []
        for notice in self._hook_dispatcher(updated).dispatch(event):
            data = {"event": notice.event, "message": notice.message}
            updated = self.store.append_runtime_event(
                updated, kind=RuntimeEventKind.HOOK_TRIGGERED.value, data=data
            )
            hook_events.append(RuntimeEvent(RuntimeEventKind.HOOK_TRIGGERED, data))
        self._session_metadata[updated.session_id] = updated
        if emit is not None:
            maybe_awaitable = emit(event)
            if maybe_awaitable is not None:
                await maybe_awaitable
            for hook_event in hook_events:
                maybe_awaitable = emit(hook_event)
                if maybe_awaitable is not None:
                    await maybe_awaitable
        return updated

    def _skill_registry(self, metadata: SessionMetadata) -> SkillRegistry:
        return SkillRegistry(
            metadata.project_root / ".repopilot" / "skills",
            additional_roots=(self.store.root / "skills",),
            plugin_roots=plugin_skill_roots(metadata.project_root, self.store.root),
        )

    def _selected_skills(self, metadata: SessionMetadata, prompt: str) -> tuple[Skill, ...]:
        registry = self._skill_registry(metadata)
        names = self.store.load_active_skills(metadata)
        explicit = tuple(skill for name in names if (skill := registry.get(name)) is not None)
        automatic = registry.select(prompt)
        return tuple(dict.fromkeys((*explicit, *automatic)))

    @staticmethod
    def _sync_state_plan(state: AgentState, plan: SessionPlan) -> None:
        state.plan = plan.render_for_context()
        state.completed_steps = [
            item.content for item in plan.todos if item.status is TodoStatus.COMPLETED
        ]

    @staticmethod
    def _content(response: ModelResponse) -> str:
        fallback = "Model returned an empty final response; please retry the request."
        try:
            raw = json.loads(response.content)
        except json.JSONDecodeError:
            return response.content.strip() or fallback
        if isinstance(raw, dict) and raw.get("type") == "finish":
            return str(raw.get("answer", "")).strip() or fallback
        return response.content.strip() or fallback

    @staticmethod
    def _turn_limit(
        state: AgentState,
        *,
        started_iteration: int,
        started_tools: int,
        started_tokens: int,
    ) -> str | None:
        if state.iteration - started_iteration >= 12:
            return "turn iteration budget exceeded"
        if state.tool_calls - started_tools >= 32:
            return "turn tool-call budget exceeded"
        if state.input_tokens + state.output_tokens - started_tokens >= 32_000:
            return "turn token budget exceeded"
        return None
