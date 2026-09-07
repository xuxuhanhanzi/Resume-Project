"""Small, explicit manager for workspace-local background processes."""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from repopilot.tools.shell import validate_tokenized_command

_MAX_CAPTURE_BYTES = 64_000


@dataclass(frozen=True, slots=True)
class ManagedProcess:
    """Stable process identity and immutable launch information."""

    process_id: str
    command: tuple[str, ...]
    cwd: Path
    started_at: datetime


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    """A bounded observation of a managed background process."""

    process: ManagedProcess
    return_code: int | None
    stdout: str
    stderr: str

    @property
    def running(self) -> bool:
        return self.return_code is None


@dataclass(slots=True)
class _Record:
    metadata: ManagedProcess
    process: asyncio.subprocess.Process
    stdout: bytearray
    stderr: bytearray
    readers: tuple[asyncio.Task[None], asyncio.Task[None]]


class ProcessManager:
    """Manage a small set of tokenized, workspace-scoped child processes.

    This primitive does not provide a model-facing tool by itself. A future tool must
    still pass through the permission boundary before calling ``start``.
    """

    def __init__(self, workspace: Path, *, max_processes: int = 4) -> None:
        self.workspace = workspace.resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"workspace does not exist: {self.workspace}")
        if max_processes <= 0:
            raise ValueError("max_processes must be positive")
        self.max_processes = max_processes
        self._records: dict[str, _Record] = {}

    async def start(self, command: Sequence[str], *, cwd: Path | None = None) -> ManagedProcess:
        """Start an argv command without a shell and begin bounded output capture."""
        running = sum(record.process.returncode is None for record in self._records.values())
        if running >= self.max_processes:
            raise RuntimeError(f"at most {self.max_processes} background processes may run")
        safe_command = validate_tokenized_command(list(command))
        selected_cwd = (cwd or self.workspace).resolve()
        if selected_cwd != self.workspace and self.workspace not in selected_cwd.parents:
            raise ValueError("background process cwd escaped its workspace")
        if not selected_cwd.is_dir():
            raise ValueError("background process cwd must be an existing directory")
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = await asyncio.create_subprocess_exec(
            *safe_command,
            cwd=str(selected_cwd),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creationflags,
        )
        metadata = ManagedProcess(uuid4().hex, safe_command, selected_cwd, datetime.now(UTC))
        stdout = bytearray()
        stderr = bytearray()
        readers = (
            asyncio.create_task(self._collect(process.stdout, stdout)),
            asyncio.create_task(self._collect(process.stderr, stderr)),
        )
        self._records[metadata.process_id] = _Record(metadata, process, stdout, stderr, readers)
        return metadata

    async def snapshot(self, process_id: str) -> ProcessSnapshot:
        """Return currently captured output and the process return code, if any."""
        record = self._record(process_id)
        if record.process.returncode is not None:
            await asyncio.gather(*record.readers)
        return ProcessSnapshot(
            record.metadata,
            record.process.returncode,
            self._decode(record.stdout),
            self._decode(record.stderr),
        )

    async def terminate(self, process_id: str, *, grace_seconds: float = 3.0) -> ProcessSnapshot:
        """Stop one known process, escalating only from graceful terminate to kill."""
        record = self._record(process_id)
        await self._stop(record.process, grace_seconds=grace_seconds)
        await asyncio.gather(*record.readers)
        return await self.snapshot(process_id)

    async def aclose(self) -> tuple[ProcessSnapshot, ...]:
        """Terminate every child owned by this manager before a session exits."""
        snapshots = await asyncio.gather(
            *(self.terminate(process_id) for process_id in tuple(self._records)),
            return_exceptions=True,
        )
        failures = [item for item in snapshots if isinstance(item, Exception)]
        if failures:
            raise RuntimeError(f"failed to clean up {len(failures)} managed process(es)")
        return tuple(item for item in snapshots if isinstance(item, ProcessSnapshot))

    def known(self) -> tuple[ManagedProcess, ...]:
        """Return launch metadata without exposing mutable subprocess handles."""
        return tuple(record.metadata for record in self._records.values())

    def _record(self, process_id: str) -> _Record:
        try:
            return self._records[process_id]
        except KeyError as error:
            raise ValueError(f"unknown process: {process_id}") from error

    @staticmethod
    async def _stop(process: asyncio.subprocess.Process, *, grace_seconds: float) -> None:
        if process.returncode is not None:
            return
        if os.name == "nt" and process.pid is not None:
            terminator = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(terminator.wait(), timeout=grace_seconds)
            except TimeoutError:
                terminator.kill()
                await terminator.wait()
            try:
                await asyncio.wait_for(process.wait(), timeout=grace_seconds)
                return
            except TimeoutError:
                pass
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        except TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                return
            await process.wait()

    @staticmethod
    async def _collect(stream: asyncio.StreamReader | None, buffer: bytearray) -> None:
        if stream is None:
            return
        while chunk := await stream.read(4_096):
            remaining = _MAX_CAPTURE_BYTES - len(buffer)
            if remaining > 0:
                buffer.extend(chunk[:remaining])

    @staticmethod
    def _decode(buffer: bytearray) -> str:
        suffix = b"\n...[output truncated by RepoPilot]"
        if len(buffer) >= _MAX_CAPTURE_BYTES:
            return (bytes(buffer[: _MAX_CAPTURE_BYTES - len(suffix)]) + suffix).decode(
                "utf-8", errors="replace"
            )
        return bytes(buffer).decode("utf-8", errors="replace")
