"""An explicit local supervisor for durable, user-submitted argv jobs.

This is intentionally separate from a conversational session.  A foreground
``repopilot supervisor serve`` process owns its children, so jobs survive an
interactive CLI closing; a service restart marks a previously running job as
``interrupted`` and never replays it.  Jobs are started only by explicit
``submit`` and ``serve`` commands, use shell-free argv, keep their working
directory inside the chosen project, and redact durable command/log evidence.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import uuid4

from repopilot.security.redaction import redact_text
from repopilot.tools.shell import validate_tokenized_command

_STATUSES = {"queued", "running", "succeeded", "failed", "stopped", "interrupted"}
_TERMINAL_STATUSES = {"succeeded", "failed", "stopped", "interrupted"}
_MAX_LOG_BYTES = 128_000
_MAX_QUEUED_JOBS = 20
_MAX_LABEL_CHARACTERS = 80
_MAX_FAILURE_REASON_CHARACTERS = 500
_MIN_TIMEOUT_SECONDS = 0.1
_MAX_TIMEOUT_SECONDS = 3_600.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class SupervisorJob:
    """One local-only job receipt, deliberately absent provider configuration."""

    job_id: str
    project_root: str
    cwd: str
    command: tuple[str, ...]
    created_at: str
    updated_at: str
    status: str = "queued"
    return_code: int | None = None
    process_id: int | None = None
    stop_requested: bool = False
    retry_of: str | None = None
    label: str = ""
    timeout_seconds: float = 900.0
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.job_id or len(self.job_id) > 64:
            raise ValueError("supervisor job ID is invalid")
        if self.status not in _STATUSES:
            raise ValueError("supervisor job status is invalid")
        validate_tokenized_command(list(self.command))
        if not self.project_root or not self.cwd or self.cwd.startswith(("/", "\\")):
            raise ValueError("supervisor job path is invalid")
        if self.return_code is not None and isinstance(self.return_code, bool):
            raise ValueError("supervisor job return code is invalid")
        if self.process_id is not None and (
            isinstance(self.process_id, bool) or self.process_id <= 0
        ):
            raise ValueError("supervisor job process ID is invalid")
        if (
            len(self.label) > _MAX_LABEL_CHARACTERS
            or "\x00" in self.label
            or "\n" in self.label
            or "\r" in self.label
            or redact_text(self.label) != self.label
        ):
            raise ValueError("supervisor job label is invalid or appears to contain a credential")
        if not _MIN_TIMEOUT_SECONDS <= self.timeout_seconds <= _MAX_TIMEOUT_SECONDS:
            raise ValueError(
                "supervisor timeout must be "
                f"{_MIN_TIMEOUT_SECONDS:g}-{_MAX_TIMEOUT_SECONDS:g} seconds"
            )
        if self.failure_reason is not None and (
            not self.failure_reason.strip()
            or len(self.failure_reason) > _MAX_FAILURE_REASON_CHARACTERS
        ):
            raise ValueError("supervisor job failure reason is invalid")
        if self.status in _TERMINAL_STATUSES and self.stop_requested:
            raise ValueError("terminal supervisor job cannot remain stop-requested")

    @property
    def retryable(self) -> bool:
        return self.status in _TERMINAL_STATUSES and self.status != "succeeded"

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "project_root": self.project_root,
            "cwd": self.cwd,
            "command": list(self.command),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "return_code": self.return_code,
            "process_id": self.process_id,
            "stop_requested": self.stop_requested,
            "retry_of": self.retry_of,
            "label": self.label,
            "timeout_seconds": self.timeout_seconds,
            "failure_reason": redact_text(self.failure_reason) if self.failure_reason else None,
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, raw: object) -> SupervisorJob:
        if not isinstance(raw, dict):
            raise ValueError("supervisor job must be an object")
        legacy = {
            "job_id",
            "project_root",
            "cwd",
            "command",
            "created_at",
            "updated_at",
            "status",
            "return_code",
            "process_id",
            "stop_requested",
            "retry_of",
            "retryable",
        }
        expected = {*legacy, "label", "timeout_seconds", "failure_reason"}
        if set(raw) != legacy and set(raw) != expected:
            raise ValueError("supervisor job contains unsupported or missing fields")
        command = raw["command"]
        scalar_values = (
            raw["job_id"],
            raw["project_root"],
            raw["cwd"],
            raw["created_at"],
            raw["updated_at"],
            raw["status"],
        )
        if (
            not all(isinstance(value, str) for value in scalar_values)
            or not isinstance(command, list)
            or not all(isinstance(value, str) for value in command)
            or not isinstance(raw["stop_requested"], bool)
            or not isinstance(raw["retryable"], bool)
            or (raw["retry_of"] is not None and not isinstance(raw["retry_of"], str))
            or ("label" in raw and not isinstance(raw["label"], str))
            or (
                "timeout_seconds" in raw
                and (
                    not isinstance(raw["timeout_seconds"], int | float)
                    or isinstance(raw["timeout_seconds"], bool)
                )
            )
            or (
                "failure_reason" in raw
                and raw["failure_reason"] is not None
                and not isinstance(raw["failure_reason"], str)
            )
        ):
            raise ValueError("supervisor job has invalid field types")
        return_code = raw["return_code"]
        process_id = raw["process_id"]
        if (
            return_code is not None
            and (not isinstance(return_code, int) or isinstance(return_code, bool))
        ) or (
            process_id is not None
            and (not isinstance(process_id, int) or isinstance(process_id, bool))
        ):
            raise ValueError("supervisor job has invalid numeric fields")
        job = cls(
            job_id=str(raw["job_id"]),
            project_root=str(raw["project_root"]),
            cwd=str(raw["cwd"]),
            command=tuple(command),
            created_at=str(raw["created_at"]),
            updated_at=str(raw["updated_at"]),
            status=str(raw["status"]),
            return_code=return_code,
            process_id=process_id,
            stop_requested=bool(raw["stop_requested"]),
            retry_of=str(raw["retry_of"]) if raw["retry_of"] is not None else None,
            label=str(raw.get("label", "")),
            timeout_seconds=float(raw.get("timeout_seconds", 900.0)),
            failure_reason=(
                str(raw["failure_reason"]) if raw.get("failure_reason") is not None else None
            ),
        )
        if bool(raw["retryable"]) != job.retryable:
            raise ValueError("supervisor job retryable state is inconsistent")
        return job


class SupervisorLease:
    """One foreground supervisor ownership marker with conservative stale handling."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._descriptor: int | None = None

    def __enter__(self) -> SupervisorLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            if self._remove_if_owner_is_definitely_gone():
                return self.__enter__()
            raise RuntimeError(
                "a RepoPilot supervisor is already running for this state root"
            ) from error
        self._descriptor = descriptor
        try:
            os.write(
                descriptor, json.dumps({"pid": os.getpid(), "created_at": _now()}).encode("utf-8")
            )
        except OSError:
            os.close(descriptor)
            self._descriptor = None
            self.path.unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        self.path.unlink(missing_ok=True)

    def _remove_if_owner_is_definitely_gone(self) -> bool:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        process_id = raw.get("pid") if isinstance(raw, dict) else None
        if not isinstance(process_id, int) or process_id <= 0 or process_id == os.getpid():
            return False
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            self.path.unlink(missing_ok=True)
            return True
        except (OSError, PermissionError):
            return False
        return False


class SupervisorStore:
    """Atomic, per-job state files suitable for a local service and CLI clients."""

    def __init__(self, state_root: Path, *, max_queued_jobs: int = _MAX_QUEUED_JOBS) -> None:
        if not 1 <= max_queued_jobs <= 1_000:
            raise ValueError("supervisor queued-job limit must be between 1 and 1000")
        self.root = state_root.resolve() / "supervisor"
        self.jobs_root = self.root / "jobs"
        self.logs_root = self.root / "logs"
        self.max_queued_jobs = max_queued_jobs

    def submit(
        self,
        *,
        project_root: Path,
        cwd: Path,
        command: tuple[str, ...],
        label: str = "",
        timeout_seconds: float = 900.0,
    ) -> SupervisorJob:
        """Persist a newly authorized shell-free job; no child starts here."""

        project = project_root.resolve(strict=True)
        selected_cwd = cwd.resolve(strict=True)
        if project != selected_cwd and project not in selected_cwd.parents:
            raise ValueError("supervisor cwd must stay within the current project")
        safe_command = validate_tokenized_command(list(command))
        if tuple(redact_text(item) for item in safe_command) != safe_command:
            raise ValueError("refusing to persist a command that appears to contain a credential")
        queued = sum(job.status == "queued" for job in self.list(project_root=project))
        if queued >= self.max_queued_jobs:
            raise ValueError(
                f"supervisor queue is full for this project ({self.max_queued_jobs} queued jobs)"
            )
        timestamp = _now()
        job = SupervisorJob(
            uuid4().hex,
            str(project),
            selected_cwd.relative_to(project).as_posix() or ".",
            safe_command,
            timestamp,
            timestamp,
            label=label.strip(),
            timeout_seconds=timeout_seconds,
        )
        self._write(job)
        return job

    def list(self, *, project_root: Path | None = None) -> tuple[SupervisorJob, ...]:
        if not self.jobs_root.is_dir():
            return ()
        expected_project = str(project_root.resolve()) if project_root is not None else None
        jobs: list[SupervisorJob] = []
        for path in self.jobs_root.glob("*.json"):
            try:
                job = SupervisorJob.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if expected_project is None or job.project_root == expected_project:
                jobs.append(job)
        return tuple(sorted(jobs, key=lambda item: item.updated_at, reverse=True))

    def resolve(self, identifier: str, *, project_root: Path | None = None) -> SupervisorJob:
        normalized = identifier.strip()
        matches = [
            job for job in self.list(project_root=project_root) if job.job_id.startswith(normalized)
        ]
        if len(matches) != 1:
            raise ValueError("supervisor job ID is unknown or ambiguous; use `supervisor list`")
        return matches[0]

    def request_stop(self, job: SupervisorJob) -> SupervisorJob:
        """Stop a queued job immediately or signal the active supervisor to stop one child."""

        current = self.resolve(job.job_id)
        if current.status in _TERMINAL_STATUSES:
            raise ValueError(f"supervisor job is already {current.status}")
        if current.status == "queued":
            updated = replace(
                current,
                status="stopped",
                stop_requested=False,
                updated_at=_now(),
            )
        else:
            updated = replace(current, stop_requested=True, updated_at=_now())
        self._write(updated)
        return updated

    def retry(self, job: SupervisorJob) -> SupervisorJob:
        current = self.resolve(job.job_id)
        if not current.retryable:
            raise ValueError("only a terminal, non-successful supervisor job can be retried")
        timestamp = _now()
        retried = SupervisorJob(
            uuid4().hex,
            current.project_root,
            current.cwd,
            current.command,
            timestamp,
            timestamp,
            retry_of=current.job_id,
            label=current.label,
            timeout_seconds=current.timeout_seconds,
        )
        self._write(retried)
        return retried

    def mark_interrupted(self, *, project_root: Path | None = None) -> tuple[SupervisorJob, ...]:
        """Make restart semantics explicit: unknown former children are never replayed."""

        interrupted: list[SupervisorJob] = []
        for job in self.list(project_root=project_root):
            if job.status != "running":
                continue
            updated = replace(
                job,
                status="interrupted",
                process_id=None,
                stop_requested=False,
                failure_reason="supervisor process was not running at restart",
                updated_at=_now(),
            )
            self._write(updated)
            interrupted.append(updated)
        return tuple(interrupted)

    def append_log(self, job_id: str, text: str) -> None:
        """Append bounded, redacted textual output for exactly one job."""

        self.logs_root.mkdir(parents=True, exist_ok=True)
        path = self.logs_root / f"{job_id}.log"
        try:
            existing = path.stat().st_size if path.exists() else 0
        except OSError:
            return
        if existing >= _MAX_LOG_BYTES:
            return
        safe = redact_text(text).encode("utf-8", errors="replace")
        remaining = _MAX_LOG_BYTES - existing
        if len(safe) > remaining:
            safe = safe[:remaining] + b"\n...[output truncated by RepoPilot supervisor]\n"
        try:
            with path.open("ab") as stream:
                stream.write(safe)
        except OSError:
            return

    def read_log(self, job: SupervisorJob) -> str:
        path = self.logs_root / f"{job.job_id}.log"
        try:
            return redact_text(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return ""

    def lease(self) -> SupervisorLease:
        return SupervisorLease(self.root / "supervisor.lock")

    def _write(self, job: SupervisorJob) -> None:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        destination = self.jobs_root / f"{job.job_id}.json"
        temporary = self.jobs_root / f".{job.job_id}.{uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps(job.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


class LocalSupervisor:
    """Run queued jobs serially in an explicit foreground supervisor process."""

    def __init__(self, store: SupervisorStore, *, project_root: Path) -> None:
        self.store = store
        self.project_root = project_root.resolve(strict=True)

    async def serve(
        self,
        *,
        poll_seconds: float = 0.2,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        if not 0.05 <= poll_seconds <= 5.0:
            raise ValueError("supervisor poll interval must be between 0.05 and 5 seconds")
        with self.store.lease():
            interrupted = self.store.mark_interrupted(project_root=self.project_root)
            if interrupted:
                print(
                    f"Marked {len(interrupted)} formerly running job(s) interrupted; "
                    "none were replayed."
                )
            print(
                "RepoPilot supervisor started. Press Ctrl+C to stop the supervisor; "
                "queued jobs wait for it."
            )
            while stop_event is None or not stop_event.is_set():
                queued = next(
                    (
                        job
                        for job in self.store.list(project_root=self.project_root)
                        if job.status == "queued"
                    ),
                    None,
                )
                if queued is None:
                    await asyncio.sleep(poll_seconds)
                    continue
                await self._run_job(queued, poll_seconds=poll_seconds)

    async def _run_job(self, submitted: SupervisorJob, *, poll_seconds: float) -> None:
        job = self.store.resolve(submitted.job_id)
        if job.status != "queued":
            return
        project = Path(job.project_root).resolve(strict=True)
        cwd = (project / job.cwd).resolve(strict=True)
        if cwd != project and project not in cwd.parents:
            raise ValueError("persisted supervisor job cwd escaped its project")
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            process = await asyncio.create_subprocess_exec(
                *job.command,
                cwd=str(cwd),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                creationflags=creationflags,
            )
        except (OSError, ValueError) as error:
            reason = redact_text(f"could not start job: {error}")[:_MAX_FAILURE_REASON_CHARACTERS]
            failed = replace(job, status="failed", failure_reason=reason, updated_at=_now())
            self.store._write(failed)
            self.store.append_log(failed.job_id, reason + "\n")
            return
        running = replace(job, status="running", process_id=process.pid, updated_at=_now())
        self.store._write(running)
        reader = asyncio.create_task(self._collect_output(running.job_id, process.stdout))
        stopped = False
        timed_out = False
        return_code = -1
        deadline = monotonic() + running.timeout_seconds
        try:
            while process.returncode is None:
                current = self.store.resolve(running.job_id)
                if current.stop_requested:
                    stopped = True
                    await self._terminate(process)
                    break
                if monotonic() >= deadline:
                    timed_out = True
                    await self._terminate(process)
                    break
                await asyncio.sleep(poll_seconds)
            return_code = await process.wait()
        except asyncio.CancelledError:
            await self._terminate(process)
            return_code = await process.wait()
            stopped = True
            raise
        finally:
            await reader
            current = self.store.resolve(running.job_id)
            failure_reason = (
                f"job timed out after {running.timeout_seconds:g} seconds" if timed_out else None
            )
            final = replace(
                current,
                status="stopped"
                if stopped or current.stop_requested
                else ("succeeded" if return_code == 0 and not timed_out else "failed"),
                return_code=return_code,
                process_id=None,
                stop_requested=False,
                failure_reason=failure_reason,
                updated_at=_now(),
            )
            self.store._write(final)

    async def _collect_output(self, job_id: str, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while chunk := await stream.read(4_096):
            self.store.append_log(job_id, chunk.decode("utf-8", errors="replace"))

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if os.name == "nt" and process.pid is not None:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except TimeoutError:
            process.kill()
            await process.wait()
