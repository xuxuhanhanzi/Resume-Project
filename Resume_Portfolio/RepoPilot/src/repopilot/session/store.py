"""Append-only session transcripts and atomic checkpoint storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from uuid import UUID, uuid4

from repopilot.core.contracts import JSONValue, Message, ToolResult
from repopilot.memory.store import SessionMemory
from repopilot.runtime.checkpoint import CheckpointStore, ExecutionJournal
from repopilot.runtime.policy import PermissionMode
from repopilot.security.redaction import redact_json_value
from repopilot.session.background_tasks import BackgroundTaskRecord
from repopilot.session.models import SessionMetadata
from repopilot.session.planning import SessionPlan
from repopilot.session.snapshots import SessionTurnSnapshot
from repopilot.verification.engine import VerificationReport


@dataclass(slots=True)
class SessionLease:
    """An exclusive local lease that prevents two CLI processes from writing one session."""

    path: Path
    _descriptor: int | None = None

    def __enter__(self) -> SessionLease:
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            if self._remove_stale_lock():
                return self.__enter__()
            raise RuntimeError("session is already open in another RepoPilot process") from error
        self._descriptor = descriptor
        payload = {
            "pid": os.getpid(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        try:
            os.write(descriptor, json.dumps(payload).encode("utf-8"))
        except OSError:
            os.close(descriptor)
            self._descriptor = None
            self.path.unlink(missing_ok=True)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        self.path.unlink(missing_ok=True)

    def _remove_stale_lock(self) -> bool:
        """Remove only a lock whose recorded owner process is definitely gone."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        pid = raw.get("pid") if isinstance(raw, dict) else None
        if not isinstance(pid, int) or pid <= 0:
            return False
        # A nested lease in this process is necessarily active.  Besides
        # avoiding a redundant platform-specific liveness syscall, this keeps
        # the expected "already open" error deterministic for re-entrant CLI
        # and test code.
        if pid == os.getpid():
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            self.path.unlink(missing_ok=True)
            return True
        except PermissionError:
            return False
        except OSError:
            return False
        return False


class SessionStore:
    """Store sessions per canonical project path under a local root directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def create(
        self,
        *,
        project_root: Path,
        model: str,
        permission_mode: PermissionMode,
        provider: str | None = None,
    ) -> SessionMetadata:
        metadata = SessionMetadata.create(
            session_id=str(uuid4()),
            project_root=project_root,
            model=model,
            permission_mode=permission_mode,
            provider=provider,
        )
        directory = self.session_dir(metadata.project_root, metadata.session_id)
        directory.mkdir(parents=True, exist_ok=False)
        self._write_metadata(metadata)
        return metadata

    def load(self, *, project_root: Path, session_id: str) -> SessionMetadata:
        canonical_id = self._canonical_session_id(session_id)
        path = self.session_dir(project_root, canonical_id) / "metadata.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"session does not exist: {session_id}") from error
        if not isinstance(raw, dict):
            raise ValueError("session metadata must be an object")
        metadata = SessionMetadata.from_dict(raw)
        if self._canonical_session_id(metadata.session_id) != canonical_id:
            raise ValueError("session metadata identity does not match its directory")
        if metadata.project_root != project_root.resolve():
            raise ValueError("session does not belong to this project")
        return metadata

    def latest(self, project_root: Path) -> SessionMetadata | None:
        sessions = self.list(project_root)
        return sessions[0] if sessions else None

    def list(self, project_root: Path, *, limit: int = 50) -> tuple[SessionMetadata, ...]:
        """List recent sessions for exactly one canonical project, newest first."""
        if limit <= 0:
            raise ValueError("session list limit must be positive")
        directory = self.project_dir(project_root) / "sessions"
        if not directory.is_dir():
            return ()
        candidates: list[SessionMetadata] = []
        for child in directory.iterdir():
            if not child.is_dir():
                continue
            try:
                candidates.append(self.load(project_root=project_root, session_id=child.name))
            except ValueError:
                continue
        if not candidates:
            return ()
        return tuple(sorted(candidates, key=lambda item: item.updated_at, reverse=True)[:limit])

    def resolve(self, project_root: Path, identifier: str) -> SessionMetadata:
        """Resolve a full UUID, unique UUID prefix, or exact session title."""
        normalized = identifier.strip()
        if not normalized:
            raise ValueError("session identifier must not be empty")
        try:
            return self.load(project_root=project_root, session_id=normalized)
        except ValueError:
            matches = [
                metadata
                for metadata in self.list(project_root)
                if metadata.session_id.startswith(normalized)
                or (metadata.title is not None and metadata.title == normalized)
            ]
        if len(matches) != 1:
            raise ValueError("session identifier is unknown or ambiguous; use /sessions")
        return matches[0]

    def fork(self, metadata: SessionMetadata) -> SessionMetadata:
        """Create an independent child session with copied durable state.

        The source transcript and checkpoint are never modified. The child receives
        a distinct identity, allowing experimentation without contaminating resume.
        """
        with self.lease(metadata):
            source = self.session_dir(metadata.project_root, metadata.session_id)
            if not source.is_dir():
                raise ValueError(f"session does not exist: {metadata.session_id}")
            child = SessionMetadata.create(
                session_id=str(uuid4()),
                project_root=metadata.project_root,
                model=metadata.model,
                permission_mode=metadata.permission_mode,
                provider=metadata.provider,
            )
            destination = self.session_dir(child.project_root, child.session_id)
            destination.mkdir(parents=True, exist_ok=False)
            for filename in (
                "transcript.jsonl",
                "checkpoint.json",
                "tool_journal.json",
                "baseline.json",
                "context.json",
                "verifications.jsonl",
                "turns.jsonl",
            ):
                source_path = source / filename
                if source_path.exists():
                    (destination / filename).write_bytes(source_path.read_bytes())
            self._append_event(child, "forked_from", {"session_id": metadata.session_id})
            self._write_metadata(child)
            return child

    def append_message(self, metadata: SessionMetadata, message: Message) -> SessionMetadata:
        updated = metadata.touched()
        self._append_event(updated, "message", {"message": message.to_dict()})
        self._write_metadata(updated)
        return updated

    def append_runtime_event(
        self, metadata: SessionMetadata, *, kind: str, data: dict[str, JSONValue]
    ) -> SessionMetadata:
        updated = metadata.touched()
        self._append_event(updated, kind, data)
        self._write_metadata(updated)
        return updated

    def set_model(
        self, metadata: SessionMetadata, model: str, *, provider: str | None = None
    ) -> SessionMetadata:
        """Persist an explicit interactive model switch and its audit record."""
        with self.lease(metadata):
            updated = metadata.with_model(model, provider=provider)
            self._append_event(
                updated,
                "model_changed",
                {"model": model, "provider": updated.provider},
            )
            self._write_metadata(updated)
            return updated

    def set_title(self, metadata: SessionMetadata, title: str) -> SessionMetadata:
        """Persist an explicit user label for a session picker or /sessions listing."""
        with self.lease(metadata):
            updated = metadata.with_title(title)
            self._append_event(updated, "session_renamed", {"title": updated.title})
            self._write_metadata(updated)
            return updated

    def recent_user_messages(
        self, metadata: SessionMetadata, *, limit: int = 20
    ) -> tuple[str, ...]:
        """Return bounded prompt history from one session without rendering tool output."""
        if limit <= 0:
            raise ValueError("history limit must be positive")
        state = self.checkpoint(metadata).load()
        if state is None:
            return ()
        prompts = [message.content for message in state.messages if message.role == "user"]
        return tuple(prompts[-limit:])

    def load_project_memory(self, project_root: Path) -> SessionMemory:
        """Load explicit, user-owned project facts independently of any one session."""
        path = self.project_dir(project_root) / "memory.json"
        if not path.exists():
            return SessionMemory()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid project memory at {path}: {error}") from error
        facts = raw.get("facts", []) if isinstance(raw, dict) else []
        if not isinstance(facts, list) or not all(isinstance(item, str) for item in facts):
            raise ValueError("project memory facts must be a list of strings")
        return SessionMemory(facts=list(facts[-20:]))

    def remember_project_fact(self, project_root: Path, fact: str) -> SessionMemory:
        """Save one bounded fact only after an explicit /remember request."""
        memory = self.load_project_memory(project_root)
        memory.remember(fact)
        self._atomic_write(
            self.project_dir(project_root) / "memory.json",
            {"facts": memory.facts},
        )
        return memory

    def checkpoint(self, metadata: SessionMetadata) -> CheckpointStore:
        path = self.session_dir(metadata.project_root, metadata.session_id) / "checkpoint.json"
        return CheckpointStore(path)

    def journal(self, metadata: SessionMetadata) -> ExecutionJournal:
        return ExecutionJournal(
            self.session_dir(metadata.project_root, metadata.session_id) / "tool_journal.json"
        )

    def baseline_path(self, metadata: SessionMetadata) -> Path:
        return self.session_dir(metadata.project_root, metadata.session_id) / "baseline.json"

    def next_turn_snapshot_sequence(self, metadata: SessionMetadata) -> int:
        """Allocate the next immutable turn-revision number without rewriting history."""

        snapshots = self.load_turn_snapshots(metadata, limit=10_000)
        return snapshots[-1].sequence + 1 if snapshots else 1

    def save_turn_snapshot(
        self, metadata: SessionMetadata, snapshot: SessionTurnSnapshot
    ) -> SessionMetadata:
        """Append one private, conflict-aware text revision after a completed operation."""

        expected = self.next_turn_snapshot_sequence(metadata)
        if snapshot.sequence != expected:
            raise ValueError("turn snapshot sequence is not the next session revision")
        updated = metadata.touched()
        path = self.session_dir(updated.project_root, updated.session_id) / "turns.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(snapshot.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        self._append_event(
            updated,
            "turn_snapshot_saved",
            {
                "sequence": snapshot.sequence,
                "kind": snapshot.kind,
                "changed_files": len(snapshot.files),
                "rewindable": snapshot.rewindable,
            },
        )
        self._write_metadata(updated)
        return updated

    def load_turn_snapshots(
        self, metadata: SessionMetadata, *, limit: int = 100
    ) -> tuple[SessionTurnSnapshot, ...]:
        """Load a bounded chronological revision history without touching workspace files."""

        if not 1 <= limit <= 10_000:
            raise ValueError("turn snapshot limit must be 1-10000")
        path = self.session_dir(metadata.project_root, metadata.session_id) / "turns.jsonl"
        if not path.exists():
            return ()
        snapshots: list[SessionTurnSnapshot] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line)
                if isinstance(raw, dict):
                    snapshots.append(SessionTurnSnapshot.from_dict(raw))
            except (ValueError, json.JSONDecodeError):
                continue
        return tuple(snapshots[-limit:])

    def load_context_summary(self, metadata: SessionMetadata) -> str | None:
        """Load the optional compact handoff summary for an existing session."""
        path = self.session_dir(metadata.project_root, metadata.session_id) / "context.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("summary"), str):
            raise ValueError("session context summary must contain a string summary")
        return str(raw["summary"])

    def save_context_summary(self, metadata: SessionMetadata, summary: str) -> SessionMetadata:
        """Atomically persist a compact summary and append an auditable transcript event."""
        if not summary.strip():
            raise ValueError("context summary must not be empty")
        updated = metadata.touched()
        self._atomic_write(
            self.session_dir(updated.project_root, updated.session_id) / "context.json",
            {"summary": summary},
        )
        self._append_event(updated, "context_compacted", {"characters": len(summary)})
        self._write_metadata(updated)
        return updated

    def restore_context_summary(
        self, metadata: SessionMetadata, summary: str | None
    ) -> SessionMetadata:
        """Restore a recorded compacted-context state during an explicit session rewind."""

        updated = metadata.touched()
        path = self.session_dir(updated.project_root, updated.session_id) / "context.json"
        if summary is None:
            path.unlink(missing_ok=True)
        else:
            self._atomic_write(path, {"summary": summary})
        self._append_event(updated, "context_restored", {"present": summary is not None})
        self._write_metadata(updated)
        return updated

    def load_active_skills(self, metadata: SessionMetadata) -> tuple[str, ...]:
        """Return explicitly activated project skills for a durable session."""
        path = self.session_dir(metadata.project_root, metadata.session_id) / "skills.json"
        if not path.exists():
            return ()
        raw = json.loads(path.read_text(encoding="utf-8"))
        names = raw.get("active", []) if isinstance(raw, dict) else []
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("session skill state must contain an active string list")
        return tuple(names)

    def persist_large_tool_result(
        self, metadata: SessionMetadata, result: ToolResult, *, inline_limit: int = 24_000
    ) -> ToolResult:
        """Store oversized text observations locally and pass a bounded reference onward.

        The model sees enough output to reason about the failure plus a stable
        path reference.  The session transcript remains redacted through the
        normal event writer; neither UI nor model output is flooded with a test
        log or build artifact.
        """

        if inline_limit < 1_000 or not isinstance(result.data, dict):
            return result
        large = {
            str(key): value
            for key, value in result.data.items()
            if isinstance(value, str) and len(value) > inline_limit
        }
        if not large:
            return result
        safe_call = re.sub(r"[^A-Za-z0-9_.-]", "_", result.call_id)[:80] or "tool"
        safe_tool = re.sub(r"[^A-Za-z0-9_.-]", "_", result.tool_name)[:80] or "result"
        relative = Path("tool-outputs") / f"{safe_call}_{safe_tool}.json"
        destination = self.session_dir(metadata.project_root, metadata.session_id) / relative
        self._atomic_write(destination, {"tool": result.tool_name, "data": large})
        trimmed = dict(result.data)
        for key, value in large.items():
            trimmed[key] = (
                value[:inline_limit]
                + f"\n...[truncated; complete redacted output: {relative.as_posix()}]"
            )
        trimmed["persisted_output_path"] = relative.as_posix()
        trimmed["persisted_output_size"] = sum(
            len(value.encode("utf-8")) for value in large.values()
        )
        return ToolResult(
            result.call_id,
            result.tool_name,
            result.ok,
            trimmed,
            result.error,
            result.error_type,
            result.recoverable,
            result.side_effect,
            result.cached,
        )

    def runtime_events(
        self, metadata: SessionMetadata, *, limit: int = 200
    ) -> tuple[dict[str, JSONValue], ...]:
        """Read bounded non-message transcript events for local diagnostics."""

        if limit <= 0:
            raise ValueError("event limit must be positive")
        path = self.session_dir(metadata.project_root, metadata.session_id) / "transcript.jsonl"
        if not path.exists():
            return ()
        events: list[dict[str, JSONValue]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict) or raw.get("kind") == "message":
                continue
            events.append(raw)
        return tuple(events[-limit:])

    def save_active_skills(
        self, metadata: SessionMetadata, names: tuple[str, ...]
    ) -> SessionMetadata:
        """Atomically persist selected skill names and record the explicit user action."""
        updated = metadata.touched()
        self._atomic_write(
            self.session_dir(updated.project_root, updated.session_id) / "skills.json",
            {"active": list(names)},
        )
        self._append_event(updated, "skills_updated", {"active": list(names)})
        self._write_metadata(updated)
        return updated

    def load_plan(self, metadata: SessionMetadata) -> SessionPlan | None:
        """Load the optional durable plan for exactly one session."""

        path = self.session_dir(metadata.project_root, metadata.session_id) / "plan.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid session plan: {error}") from error
        if not isinstance(raw, dict):
            raise ValueError("session plan must be an object")
        return SessionPlan.from_dict(raw)

    def save_plan(self, metadata: SessionMetadata, plan: SessionPlan) -> SessionMetadata:
        """Persist a human-created or model-maintained plan and record its status."""

        updated = metadata.touched()
        self._atomic_write(
            self.session_dir(updated.project_root, updated.session_id) / "plan.json",
            plan.to_dict(),
        )
        self._append_event(
            updated,
            "plan_updated",
            {"approved": plan.approved, "todo_count": len(plan.todos)},
        )
        self._write_metadata(updated)
        return updated

    def save_verification_report(
        self, metadata: SessionMetadata, report: VerificationReport
    ) -> SessionMetadata:
        """Append one complete, redacted verification report to its session history."""

        updated = metadata.touched()
        path = self.session_dir(updated.project_root, updated.session_id) / "verifications.jsonl"
        payload = redact_json_value(report.to_dict())
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self._append_event(
            updated,
            "verification_report_saved",
            {"ok": report.ok, "result_count": len(report.results)},
        )
        self._write_metadata(updated)
        return updated

    def load_background_tasks(self, metadata: SessionMetadata) -> tuple[BackgroundTaskRecord, ...]:
        """Read durable task receipts; no process is started or reattached."""

        path = (
            self.session_dir(metadata.project_root, metadata.session_id) / "background_tasks.json"
        )
        if not path.exists():
            return ()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid background task receipts: {error}") from error
        items = raw.get("tasks") if isinstance(raw, dict) else None
        if not isinstance(items, list) or len(items) > 100:
            raise ValueError("background task receipts must contain at most 100 tasks")
        tasks = tuple(BackgroundTaskRecord.from_dict(item) for item in items)
        if len({item.process_id for item in tasks}) != len(tasks):
            raise ValueError("background task receipt IDs must be unique")
        return tasks

    def save_background_tasks(
        self, metadata: SessionMetadata, tasks: tuple[BackgroundTaskRecord, ...]
    ) -> SessionMetadata:
        """Persist redacted task receipts without output streams or command replay."""

        if len(tasks) > 100:
            raise ValueError("background task receipts must contain at most 100 tasks")
        updated = metadata.touched()
        self._atomic_write(
            self.session_dir(updated.project_root, updated.session_id) / "background_tasks.json",
            {"tasks": [item.to_dict() for item in tasks]},
        )
        self._append_event(
            updated,
            "background_tasks_updated",
            {
                "count": len(tasks),
                "running": sum(item.status == "running" for item in tasks),
            },
        )
        self._write_metadata(updated)
        return updated

    def load_verification_reports(
        self, metadata: SessionMetadata, *, limit: int = 10
    ) -> tuple[VerificationReport, ...]:
        """Read a bounded newest-first verification history without executing commands."""

        if not 1 <= limit <= 100:
            raise ValueError("verification history limit must be 1-100")
        path = self.session_dir(metadata.project_root, metadata.session_id) / "verifications.jsonl"
        if not path.exists():
            return ()
        reports: list[VerificationReport] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    continue
                reports.append(VerificationReport.from_dict(raw))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return tuple(reversed(reports[-limit:]))

    def project_dir(self, project_root: Path) -> Path:
        canonical = str(project_root.resolve()).encode("utf-8")
        return self.root / "projects" / hashlib.sha256(canonical).hexdigest()[:24]

    def session_dir(self, project_root: Path, session_id: str) -> Path:
        return self.project_dir(project_root) / "sessions" / self._canonical_session_id(session_id)

    def lease(self, metadata: SessionMetadata) -> SessionLease:
        """Acquire an exclusive lease for all mutable operations on one session."""
        return SessionLease(
            self.session_dir(metadata.project_root, metadata.session_id) / ".repopilot.lock"
        )

    @staticmethod
    def _canonical_session_id(session_id: str) -> str:
        try:
            return str(UUID(session_id))
        except (AttributeError, ValueError) as error:
            raise ValueError("session_id must be a canonical UUID") from error

    def _write_metadata(self, metadata: SessionMetadata) -> None:
        self._atomic_write(
            self.session_dir(metadata.project_root, metadata.session_id) / "metadata.json",
            metadata.to_dict(),
        )

    def _append_event(
        self, metadata: SessionMetadata, kind: str, data: dict[str, JSONValue]
    ) -> None:
        path = self.session_dir(metadata.project_root, metadata.session_id) / "transcript.jsonl"
        payload: dict[str, JSONValue] = {
            "kind": kind,
            "data": redact_json_value(data),
        }
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _atomic_write(path: Path, data: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), "utf-8"
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
